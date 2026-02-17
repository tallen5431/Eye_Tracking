# eye_ops/density_refine.py
import numpy as np
import cv2

def mask_density(mask01: np.ndarray, k: int = 31) -> np.ndarray:
    """Local dark-pixel density from a 0/1 mask. Returns uint8 0..255."""
    m = mask01.astype(np.float32)
    dens = cv2.boxFilter(m, ddepth=-1, ksize=(k, k), normalize=True)
    return (dens * 255).astype(np.uint8)

def pick_component_from_mask01(mask01: np.ndarray, *, min_area: int, max_area: int):
    """Pick best component like your existing scorer. mask01 is 0/1."""
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
                "bbox": (int(x), int(y), int(ww), int(hh)),
                "area": int(area),
                "center": (float(cx), float(cy)),
            }
    return best, labels

def fit_ellipse_from_labels(labels: np.ndarray, comp_id: int):
    ys, xs = np.where(labels == comp_id)
    if len(xs) < 20:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
    if len(pts) < 5:
        return None
    try:
        return cv2.fitEllipse(pts)
    except Exception:
        return None

def refine_best_with_density(
    mask01: np.ndarray,
    *,
    k: int = 31,
    density_thr: int = 120,
    min_area: int = 300,
    max_area: int = 250000,
):
    """
    Returns (best, ellipse, dens_u8, dens_mask01).
    dens_mask01 is the thresholded high-density binary mask used for fitting.
    """
    dens = mask_density(mask01, k=k)
    dens_mask01 = (dens >= int(density_thr)).astype(np.uint8)

    best, labels = pick_component_from_mask01(dens_mask01, min_area=min_area, max_area=max_area)
    ellipse = None
    if best is not None:
        ellipse = fit_ellipse_from_labels(labels, best["id"])
    return best, ellipse, dens, dens_mask01
