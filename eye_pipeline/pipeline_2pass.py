# eye_pipeline/pipeline_2pass.py
import time
import numpy as np
import cv2

from eye_ops import settings as S
from eye_ops.roi import crop_roi
from eye_ops.intensity import step2_intensity
from eye_ops.masks import mask_percentile, mask_percentile_bright, mask_adaptive
from eye_ops.geometry import fit_ellipse
from eye_ops.crop_map import crop_rotated_by_green_bbox, apply_padding
from eye_ops.density_refine import refine_best_with_density


# ----------------------------
# Small helpers
# ----------------------------
def downscale_with_factor(img, factor: float):
    if float(factor) == 1.0:
        return img, 1.0
    out = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return out, float(factor)


def pick_component_scaled(mask01, min_area: int, max_area: int):
    """
    Same as pick_component(), but with per-pass min/max area.
    """
    num, labels, stats_, centroids = cv2.connectedComponentsWithStats(
        (mask01 * 255).astype(np.uint8), connectivity=8
    )
    h, w = mask01.shape[:2]
    cx0, cy0 = w / 2.0, h / 2.0

    best = None
    best_score = -1e18

    for i in range(1, num):
        x, y, ww, hh, area = stats_[i]
        if area < min_area or area > max_area:
            continue
        cx, cy = centroids[i]
        aspect = max(ww / max(hh, 1), hh / max(ww, 1))
        dist = np.hypot(cx - cx0, cy - cy0)
        score = area - 40 * dist - 400 * (aspect - 1.0)
        if score > best_score:
            best_score = score
            best = {
                "id": i,
                "bbox": (x, y, ww, hh),
                "area": int(area),
                "center": (float(cx), float(cy)),
            }
    return best, labels


def _scaled_area_thresholds(downscale_factor: float, area_scale_ref_factor: float | None):
    # Coarse pass
    if area_scale_ref_factor is None:
        return int(S.MIN_AREA), int(S.MAX_AREA)

    # Fine pass override if provided
    if hasattr(S, "MIN_AREA_FINE") and hasattr(S, "MAX_AREA_FINE"):
        return int(S.MIN_AREA_FINE), int(S.MAX_AREA_FINE)

    # Fallback: scale from coarse
    ref = float(area_scale_ref_factor)
    cur = float(downscale_factor)
    mult = (cur / max(1e-9, ref)) ** 2
    return int(round(S.MIN_AREA * mult)), int(round(S.MAX_AREA * mult))


def _prefix_timings(prefix: str, t: dict) -> dict:
    return {f"{prefix}_{k}": float(v) for k, v in t.items()}


# ----------------------------
# Core pipeline for ONE ROI
# ----------------------------
def run_pipeline_core(
    img_bgr,
    *,
    roi_origin_full=(0, 0),
    downscale_factor=1.0,
    area_scale_ref_factor=None,
    draw_viz: bool = True,   # dev only; set False for production timing
):
    """
    Core pipeline on the given BGR image (already ROI-selected externally).
    Returns: result_dict, timing_dict
    """
    t = {}
    t0 = time.perf_counter()

    # Step 1: (external ROI already chosen)
    a = time.perf_counter()
    roi_bgr = img_bgr
    b = time.perf_counter()
    t["roi"] = b - a

    # Step 1b: Downscale (pass-specific)
    roi_small, scale = downscale_with_factor(roi_bgr, downscale_factor)
    c = time.perf_counter()
    t["downscale"] = c - b

    # Step 2: intensity
    gray = step2_intensity(roi_small)
    d = time.perf_counter()
    t["step2"] = d - c

    # Step 3: percentile mask
    is_fine = area_scale_ref_factor is not None  # coarse uses None, fine uses coarse ref
    
    # Choose fill-holes toggle (fine can override)
    fill_holes = getattr(S, "FILL_HOLES", True)
    if is_fine:
        fill_holes = getattr(S, "FILL_HOLES_FINE", fill_holes)
    
    if is_fine and hasattr(S, "PCT_FINE"):
        _, m_pct = mask_percentile(
            gray,
            pct=S.PCT_FINE,
            blur_k=S.BLUR_K_FINE,
            open_k=S.OPEN_K_FINE,
            close_k=S.CLOSE_K_FINE,
            fill=fill_holes,
        )
    else:
        _, m_pct = mask_percentile(gray, fill=fill_holes)

    # Step 3-glare: subtract bright glare regions from the dark pupil mask (fine pass only)
    glare_mask = None
    if is_fine and bool(getattr(S, "USE_GLARE_MASK_FINE", False)):
        _, glare_mask = mask_percentile_bright(gray)
        m_pct = np.clip(m_pct.astype(np.int16) - glare_mask.astype(np.int16), 0, 1).astype(np.uint8)

    e = time.perf_counter()
    t["threshold"] = e - d


    # Step 3b: pick blob (scale-aware)
    min_area_eff, max_area_eff = _scaled_area_thresholds(downscale_factor, area_scale_ref_factor)
    best_pct, labels_pct = pick_component_scaled(m_pct, min_area_eff, max_area_eff)
    f = time.perf_counter()
    t["pick"] = f - e

    best = best_pct
    labels = labels_pct
    chosen_mask = m_pct
    method = "percentile"
    ellipse = None            # <-- ADD THIS (prevents UnboundLocalError)
    density_map_u8 = None     # <-- optional but keeps things consistent

    # Step 3c: Density Map (fine pass only)
    if is_fine and bool(getattr(S, "USE_DENSITY_REFINE_FINE", False)):
        best_d, ell_d, dens_u8, dens_mask01 = refine_best_with_density(
            m_pct,
            k=int(getattr(S, "DENSITY_K_FINE", 31)),
            density_thr=int(getattr(S, "DENSITY_THR_FINE", 120)),
            min_area=int(min_area_eff),
            max_area=int(max_area_eff),
            ellipse_mode=str(getattr(S, "DENSITY_ELLIPSE_MODE_FINE", "weighted")),
            cover_frac=float(getattr(S, "DENSITY_COVER_FRAC_FINE", 0.90)),
        )  
        density_map_u8 = dens_u8
        chosen_mask = dens_mask01
        method = "density"
    
        if best_d is not None:
            best = best_d
        if ell_d is not None:
            ellipse = ell_d


    # Optional adaptive + repick
    if bool(S.USE_ADAPTIVE):
        m_adp = mask_adaptive(gray)
        best_adp, labels_adp = pick_component_scaled(m_adp, min_area_eff, max_area_eff)
        if best_adp is not None:
            best = best_adp
            labels = labels_adp
            chosen_mask = m_adp
            method = "adaptive"

    # Step 4: ellipse (only if we don't already have one from density refine)
    g = time.perf_counter()
    if ellipse is None and bool(S.DO_ELLIPSE) and best is not None:
        ellipse = fit_ellipse(labels, best["id"])
    h = time.perf_counter()
    t["ellipse"] = h - g


    # Step 5: viz overlay (DEV ONLY)
    viz = None
    if bool(draw_viz):
        viz0 = time.perf_counter()
        viz = roi_small.copy()
        if best is not None:
            x, y, ww, hh = best["bbox"]
            cx, cy = best["center"]
            cv2.rectangle(viz, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            cv2.circle(viz, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            if ellipse is not None:
                cv2.ellipse(viz, ellipse, (255, 255, 0), 2)
        viz1 = time.perf_counter()
        t["viz"] = viz1 - viz0
        end = viz1
    else:
        t["viz"] = 0.0
        end = time.perf_counter()

    t["total"] = end - t0

    result = {
        "roi_small": roi_small,
        "gray": gray,
        "mask_pct": m_pct,
        "glare_mask": glare_mask,
        "density_u8": density_map_u8,
        "mask_final": chosen_mask,
        "viz": viz,
        "method": method,
        "ellipse": ellipse,
        "best": best,
        "roi_origin": tuple(map(int, roi_origin_full)),
        "scale": float(downscale_factor),
        "min_area_eff": int(min_area_eff),
        "max_area_eff": int(max_area_eff),
    }
    return result, t


# ----------------------------
# 2-pass pipeline for ONE frame
# ----------------------------
def run_pipeline_on_image(rotated_full_bgr):
    """
    Returns: fine_result, timing_dict

    timing_dict includes:
      - coarse_* timings (coarse_total, coarse_step2, ...)
      - crop (map+pad+slice)
      - fine_* timings
      - total_2pass (coarse_total + crop + fine_total)
      - total_wall (wall clock for this function)
    """
    t_wall0 = time.perf_counter()

    # DEV controls (default True if not defined)
    draw_coarse = bool(getattr(S, "DRAW_VIZ_COARSE", True))
    draw_fine = bool(getattr(S, "DRAW_VIZ_FINE", True))

    # ----- PASS A (coarse) -----
    roi_bgr, (rx, ry) = crop_roi(rotated_full_bgr)
    coarse_res, coarse_t = run_pipeline_core(
        roi_bgr,
        roi_origin_full=(rx, ry),
        downscale_factor=float(S.DOWNSCALE),
        area_scale_ref_factor=None,
        draw_viz=draw_coarse,
    )

    # ----- CROP STEP (bbox map + padding + slice) -----
    t_crop0 = time.perf_counter()
    _, bb_full = crop_rotated_by_green_bbox(rotated_full_bgr, coarse_res)
    crop_bgr = None
    bb_pad = None

    if bb_full is not None:
        bb_pad = apply_padding(bb_full, rotated_full_bgr.shape, S.PAD_PX, S.PAD_REL)
        if bb_pad is not None:
            X0, Y0, X1, Y1 = bb_pad
            crop_bgr = rotated_full_bgr[Y0:Y1, X0:X1].copy()
    t_crop1 = time.perf_counter()
    crop_t = t_crop1 - t_crop0

    # If we fail to crop, fall back to coarse result as "fine"
    if crop_bgr is None:
        coarse_res["coarse"] = None
        coarse_res["crop_full_bbox"] = None
        coarse_res["crop_bgr"] = None
        coarse_res["timing_coarse"] = coarse_t

        timing = {}
        timing.update(_prefix_timings("coarse", coarse_t))
        timing["crop"] = float(crop_t)
        timing["fine_total"] = 0.0
        timing["fine_skipped"] = 1.0
        timing["total_2pass"] = float(coarse_t.get("total", 0.0)) + float(crop_t)
        timing["total_wall"] = float(time.perf_counter() - t_wall0)
        return coarse_res, timing

    # ----- PASS B (fine) -----
    X0, Y0, X1, Y1 = bb_pad
    fine_res, fine_t = run_pipeline_core(
        crop_bgr,
        roi_origin_full=(X0, Y0),
        downscale_factor=float(S.DOWNSCALE_FINE),
        area_scale_ref_factor=float(S.DOWNSCALE),
        draw_viz=draw_fine,
    )

    # Attach metadata for later mapping/debug
    fine_res["coarse"] = coarse_res
    fine_res["crop_full_bbox"] = bb_pad
    fine_res["crop_bgr"] = crop_bgr
    fine_res["timing_coarse"] = coarse_t

    timing = {}
    timing.update(_prefix_timings("coarse", coarse_t))
    timing["crop"] = float(crop_t)
    timing.update(_prefix_timings("fine", fine_t))
    timing["fine_skipped"] = 0.0
    timing["coarse_total"] = float(coarse_t.get("total", 0.0))
    timing["fine_total"] = float(fine_t.get("total", 0.0))
    timing["total_2pass"] = timing["coarse_total"] + timing["crop"] + timing["fine_total"]
    timing["total_wall"] = float(time.perf_counter() - t_wall0)
    return fine_res, timing


# ----------------------------
# Batch runner
# ----------------------------
def run_pipeline_batch(images_bgr):
    results = []
    timings = None

    t_start = time.perf_counter()
    for img in images_bgr:
        res, t = run_pipeline_on_image(img)
        results.append(res)

        if timings is None:
            timings = {k: [] for k in t.keys()}

        # Ensure stable keys even if edge cases change dict contents
        for k in timings.keys():
            timings[k].append(float(t.get(k, 0.0)))

    t_end = time.perf_counter()
    if timings is None:
        timings = {}

    return results, timings, (t_end - t_start)
