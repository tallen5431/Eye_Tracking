#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import json
from pathlib import Path
import cv2
import numpy as np

from eye_ops import settings as S
from eye_pipeline.pipeline_2pass import run_pipeline_on_image
from eye_ops.overlay import draw_pupil_overlay_on_full


# ---------------------------------------------------------------------------
# Pipeline stage / view definitions
# ---------------------------------------------------------------------------
# Each entry: (key_label, display_name, function(res, frame) -> bgr_image)
# The lambdas are filled in after the helper functions are defined.

STAGE_NAMES = [
    "frame+overlay",       # 0  f
    "coarse+overlay",      # 1  1
    "coarse mask",         # 2  2
    "coarse bright mask",  # 3  3
    "fine mask",           # 4  4
    "fine density",        # 5  5
    "glare mask",          # 6  6
    "fine frame+overlay",  # 7  7
]

HOTKEY_HELP = (
    "q/ESC=quit  space=pause  s=split  o=overlay  "
    "f=full-frame  1-7=left-panel  shift+1-7=right-panel"
)


# ---------------------------------------------------------------------------
# Atomic JSON write (unchanged)
# ---------------------------------------------------------------------------
def _atomic_write_json(path, payload):
    """Best-effort atomic JSON write on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    for attempt in range(30):
        try:
            os.replace(str(tmp), str(path))
            return
        except (PermissionError, OSError):
            time.sleep(0.005 * (attempt + 1))
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Result helpers (unchanged)
# ---------------------------------------------------------------------------
def _pick_best_result(res: dict) -> dict | None:
    if not isinstance(res, dict):
        return None
    if res.get("best") is not None:
        return res
    c = res.get("coarse")
    if isinstance(c, dict) and c.get("best") is not None:
        return c
    return None


def _pick_stage_source(res: dict) -> tuple[str, dict | None]:
    if isinstance(res, dict) and res.get("best") is not None:
        return "FINE", res
    c = res.get("coarse") if isinstance(res, dict) else None
    if isinstance(c, dict) and c.get("best") is not None:
        return "COARSE", c
    return "NONE", None


def _map_center_to_full(res_used: dict) -> tuple[float, float] | None:
    best = res_used.get("best")
    if best is None:
        return None
    cx, cy = best["center"]
    ox, oy = res_used.get("roi_origin", (0, 0))
    scale = float(res_used.get("scale", 1.0))
    return (float(cx) / scale) + float(ox), (float(cy) / scale) + float(oy)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _density_u8_from_mask(mask01, k: int = 31) -> np.ndarray:
    k = max(3, int(k) | 1)  # ensure odd and >= 3
    m = mask01.astype("float32")
    dens = cv2.boxFilter(m, ddepth=-1, ksize=(k, k), normalize=True)
    return (dens * 255.0).astype("uint8")


def _mask_to_bgr(mask01) -> np.ndarray:
    """0/1 mask → white-on-black BGR."""
    return cv2.cvtColor((mask01.astype("uint8") * 255), cv2.COLOR_GRAY2BGR)


def _gray_to_bgr(gray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _placeholder(shape_hw, text: str) -> np.ndarray:
    """Dark panel with centred text — shown when a stage has no data."""
    h, w = shape_hw
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (w // 2 - 80, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    return img


def _hud(img: np.ndarray, lines: list[str], color=(0, 255, 0)) -> np.ndarray:
    """Draw small status lines in the top-left corner (in-place copy)."""
    out = img.copy()
    for i, txt in enumerate(lines):
        cv2.putText(out, txt, (8, 22 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def _stage_badge(img: np.ndarray, label: str) -> np.ndarray:
    """Coloured badge in top-right showing FINE / COARSE / NONE."""
    colors = {"FINE": (0, 200, 0), "COARSE": (0, 180, 255), "NONE": (60, 60, 60)}
    col = colors.get(label, (120, 120, 120))
    h, w = img.shape[:2]
    tw, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0], None
    x0 = w - tw[0] - 14
    cv2.rectangle(img, (x0 - 4, 4), (w - 4, 30), col, -1)
    cv2.putText(img, label, (x0, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Per-mode rendering
# ---------------------------------------------------------------------------
def _render_mode(mode_idx: int, res: dict, frame: np.ndarray,
                 stage_label: str, show_overlay: bool,
                 density_k: int) -> np.ndarray:
    """
    Return a BGR image for the given mode index.
    mode_idx:
      0  full frame (+ overlay if enabled)
      1  coarse frame + overlay
      2  coarse dark mask (binary)
      3  coarse bright mask (binary)
      4  fine dark mask (binary)
      5  fine density (inferno colourmap)
      6  glare mask (binary)
      7  fine frame + overlay
    """
    H, W = frame.shape[:2]
    coarse = res.get("coarse") if isinstance(res, dict) else None

    # --- mode 0: full frame ---
    if mode_idx == 0:
        disp = frame.copy()
        if show_overlay:
            stage_label2, src = _pick_stage_source(res)
            if src is not None:
                disp = draw_pupil_overlay_on_full(disp, src, draw_bbox=True)
        return disp

    # --- mode 1: coarse frame + overlay ---
    if mode_idx == 1:
        disp = frame.copy()
        if coarse is not None and coarse.get("best") is not None:
            disp = draw_pupil_overlay_on_full(disp, coarse, draw_bbox=True)
        elif coarse is None:
            cv2.putText(disp, "no coarse result", (10, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 200), 2)
        return disp

    # --- mode 2: coarse dark mask ---
    if mode_idx == 2:
        src = coarse if coarse is not None else res
        m = src.get("mask_pct") if src else None
        if m is not None:
            return _mask_to_bgr(m)
        return _placeholder((H, W), "no coarse mask")

    # --- mode 3: coarse bright mask ---
    if mode_idx == 3:
        src = coarse if coarse is not None else res
        m = src.get("coarse_bright_mask") if src else None
        if m is not None:
            # Colour the bright regions orange on the coarse grayscale for context
            gray = src.get("gray")
            if gray is not None:
                base = cv2.cvtColor(
                    cv2.resize(gray, (W, H), interpolation=cv2.INTER_NEAREST),
                    cv2.COLOR_GRAY2BGR,
                )
                bright_up = cv2.resize(m.astype("uint8"),
                                       (W, H), interpolation=cv2.INTER_NEAREST)
                base[bright_up == 1] = (0, 100, 255)  # orange
                return base
            return _mask_to_bgr(m)
        if not bool(getattr(S, "USE_BRIGHT_COARSE", False)):
            return _placeholder((H, W), "USE_BRIGHT_COARSE=False (disabled)")
        return _placeholder((H, W), "no coarse bright mask")

    # --- mode 4: fine dark mask ---
    if mode_idx == 4:
        m = res.get("mask_pct") if isinstance(res, dict) else None
        if m is not None:
            return _mask_to_bgr(m)
        return _placeholder((H, W), "no fine mask")

    # --- mode 5: fine density ---
    if mode_idx == 5:
        # Prefer stored density_u8, fall back to computing from mask_pct
        d = res.get("density_u8") if isinstance(res, dict) else None
        if d is None:
            m = res.get("mask_pct") if isinstance(res, dict) else None
            if m is not None:
                d = _density_u8_from_mask(m, k=density_k)
        if d is not None:
            return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
        return _placeholder((H, W), "no fine density")

    # --- mode 6: glare mask ---
    if mode_idx == 6:
        m = res.get("glare_mask") if isinstance(res, dict) else None
        if m is not None:
            # Colour glare regions red on the fine grayscale
            gray = res.get("gray")
            if gray is not None:
                base = cv2.cvtColor(
                    cv2.resize(gray, (W, H), interpolation=cv2.INTER_NEAREST),
                    cv2.COLOR_GRAY2BGR,
                )
                glare_up = cv2.resize(m.astype("uint8"),
                                      (W, H), interpolation=cv2.INTER_NEAREST)
                base[glare_up == 1] = (0, 0, 220)  # red
                return base
            return _mask_to_bgr(m)
        if not bool(getattr(S, "USE_GLARE_MASK_FINE", False)):
            return _placeholder((H, W), "USE_GLARE_MASK_FINE=False (disabled)")
        return _placeholder((H, W), "no glare mask")

    # --- mode 7: fine frame + overlay ---
    if mode_idx == 7:
        disp = frame.copy()
        if isinstance(res, dict) and res.get("best") is not None:
            disp = draw_pupil_overlay_on_full(disp, res, draw_bbox=True)
        return disp

    return frame.copy()


def _panel_label(mode_idx: int) -> str:
    return STAGE_NAMES[mode_idx] if 0 <= mode_idx < len(STAGE_NAMES) else "?"


def _build_display(left: np.ndarray, right: np.ndarray | None,
                   left_name: str, right_name: str,
                   stage_label: str, fps: float, ms_total: float,
                   fine_skipped: bool, split: bool,
                   scale: float) -> np.ndarray:
    """Compose the final display image (single or split)."""
    if split and right is not None:
        # Resize both panels to same height then stack side-by-side
        H = left.shape[0]
        if right.shape[0] != H:
            right = cv2.resize(right, (right.shape[1], H))
        divider = np.full((H, 3, 3), (60, 60, 60), dtype=np.uint8)
        disp = np.hstack([left, divider, right])

        # Label bars at top of each panel
        lw = left.shape[1]
        rw = right.shape[1]
        cv2.rectangle(disp, (0, 0), (lw, 22), (30, 30, 30), -1)
        cv2.rectangle(disp, (lw + 3, 0), (lw + 3 + rw, 22), (30, 30, 30), -1)
        cv2.putText(disp, f"LEFT: {left_name}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(disp, f"RIGHT: {right_name}", (lw + 9, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        disp = left.copy()

    # HUD
    fine_note = " (fine skipped)" if fine_skipped else ""
    lines = [
        f"fps~{fps:5.1f}  ms={ms_total:5.1f}  stage={stage_label}{fine_note}",
        f"{'SPLIT' if split else 'SINGLE'}  {HOTKEY_HELP}",
    ]
    disp = _hud(disp, lines)
    _stage_badge(disp, stage_label)

    if scale != 1.0:
        disp = cv2.resize(disp, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    return disp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    share_path = Path(os.environ.get(
        "TRACK_SHARE_FILE", str(Path.cwd() / "logs" / "pupil_share.json")))
    cam_id = int(os.environ.get("CAMERA_ID", "0"))
    rotate_cw = os.environ.get("ROTATE_90_CW", "1") not in ("0", "false", "False")

    target_fps   = float(os.environ.get("TARGET_FPS", os.environ.get("FPS", "60")))
    write_missing = os.environ.get("WRITE_WHEN_MISSING", "0") in ("1", "true", "True")
    disable_viz  = os.environ.get("DISABLE_VIZ", "1") in ("1", "true", "True")

    cam_w   = int(os.environ.get("CAM_W", "0"))
    cam_h   = int(os.environ.get("CAM_H", "0"))
    cam_fps = float(os.environ.get("CAM_FPS", "0"))
    cam_backend = os.environ.get("CAM_BACKEND", "auto").lower()

    preview       = os.environ.get("PREVIEW", "1") in ("1", "true", "True")
    preview_scale = float(os.environ.get("PREVIEW_SCALE", "0.75"))
    window_name   = os.environ.get("PREVIEW_WINDOW", "2-pass Tracker Preview")
    density_k     = int(os.environ.get("DENSITY_K_PREVIEW", "31"))

    if disable_viz:
        S.DRAW_VIZ_COARSE = False
        S.DRAW_VIZ_FINE   = False

    # Open camera
    api = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}.get(cam_backend, 0)
    cap = cv2.VideoCapture(cam_id, api) if api else cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera id {cam_id}")

    if cam_w > 0: cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(cam_w))
    if cam_h > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cam_h))
    if cam_fps > 0: cap.set(cv2.CAP_PROP_FPS,        float(cam_fps))

    # --- view state ---
    left_mode  = 0   # index into STAGE_NAMES for left panel
    right_mode = 4   # index for right panel (fine mask by default)
    split_view = False
    show_overlay = True
    paused = False

    last_payload = None
    frame_interval = 1.0 / max(1e-6, target_fps)

    # Cache last good result so paused view stays populated
    last_res   = None
    last_frame = None

    if preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    n_modes = len(STAGE_NAMES)

    print("=" * 55)
    print("Live 2-pass tracker  —  stage-comparison preview")
    print(f" Camera:     {cam_id}  rotate_cw={rotate_cw}")
    print(f" Share file: {share_path}")
    print(f" Target FPS: {target_fps}")
    print(f" Preview:    {preview}  scale={preview_scale}")
    print()
    print(" Hotkeys:")
    print("  q / ESC      quit")
    print("  space        pause / resume")
    print("  s            toggle split / single view")
    print("  o            toggle overlay (frame modes)")
    print("  f            left panel → full frame+overlay")
    print("  1 – 7        left panel → stage 1-7")
    print("  shift+1 – 7  right panel → stage 1-7")
    print("  shift+f      right panel → full frame+overlay")
    print()
    print(" Stage indices:")
    for i, name in enumerate(STAGE_NAMES):
        print(f"   {'f' if i == 0 else i} = {name}")
    print("=" * 55)

    while True:
        t0 = time.perf_counter()

        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            if rotate_cw:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            res, timing = run_pipeline_on_image(frame)
            last_res   = res
            last_frame = frame

            used = _pick_best_result(res)
            stage_label, _ = _pick_stage_source(res)
            fine_skipped = float(timing.get("fine_skipped", 0.0) or 0.0)
            ms_total = float(1000.0 * timing.get(
                "total_2pass", timing.get("fine_total", timing.get("total", 0.0))))

            # Write share file
            if used is not None:
                center = _map_center_to_full(used)
                if center is not None:
                    X, Y = center
                    H_f, W_f = frame.shape[:2]
                    payload = {
                        "ellipse":     {"cx": float(X), "cy": float(Y)},
                        "frame_w":     int(W_f),
                        "frame_h":     int(H_f),
                        "confidence":  float(used.get("best", {}).get("area", 0) or 0.0),
                        "ts":          float(time.time()),
                        "timing_ms":   ms_total,
                        "stage":       stage_label,
                        "fine_skipped": fine_skipped,
                    }
                    _atomic_write_json(share_path, payload)
                    last_payload = payload
            else:
                if write_missing and last_payload is not None:
                    last_payload.update({
                        "ts": float(time.time()),
                        "timing_ms": ms_total,
                        "stage": stage_label,
                        "fine_skipped": fine_skipped,
                    })
                    _atomic_write_json(share_path, last_payload)

        else:
            # Paused — use cached values
            res         = last_res   or {}
            frame       = last_frame if last_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            stage_label = "PAUSED"
            ms_total    = 0.0
            fine_skipped = 0.0

        # --- render preview ---
        if preview and last_frame is not None:
            fps_est = 1.0 / max(1e-9, time.perf_counter() - t0)

            left_img = _render_mode(left_mode, res, frame,
                                    stage_label, show_overlay, density_k)
            right_img = None
            if split_view:
                right_img = _render_mode(right_mode, res, frame,
                                         stage_label, show_overlay, density_k)

            disp = _build_display(
                left_img, right_img,
                _panel_label(left_mode), _panel_label(right_mode),
                stage_label, fps_est, ms_total,
                fine_skipped > 0.5, split_view, preview_scale,
            )
            cv2.imshow(window_name, disp)

        # --- key handling ---
        if preview:
            k = cv2.waitKey(1) & 0xFF
            shift = False

            # Detect shift via key value offset (OpenCV gives shifted chars)
            # Shift+1='!' (33), shift+2='"'(34)... up to shift+7='&'(38)
            # shift+f='F' (70)
            shifted_map = {
                ord("!"): 1, ord('"'): 2, ord("#"): 3,
                ord("$"): 4, ord("%"): 5, ord("&"): 6, ord("'"): 7,
                ord("F"): 0,  # shift+f → right panel full frame
            }

            if k in (27, ord("q")):
                break
            elif k == ord(" "):
                paused = not paused
            elif k == ord("s"):
                split_view = not split_view
            elif k == ord("o"):
                show_overlay = not show_overlay
            elif k == ord("f"):
                left_mode = 0  # full frame+overlay
            elif ord("1") <= k <= ord("7"):
                left_mode = k - ord("0")   # 1-7
            elif k in shifted_map:
                right_mode = shifted_map[k]

        # FPS cap
        if not paused:
            sleep_for = frame_interval - (time.perf_counter() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)

    cap.release()
    if preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
