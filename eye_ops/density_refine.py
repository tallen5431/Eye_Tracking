# eye_ops/density_refine.py
import numpy as np
import cv2

def mask_density(mask01: np.ndarray, k: int = 31) -> np.ndarray:
    """Local dark-pixel density from a 0/1 mask. Returns uint8 0..255."""
    m = mask01.astype(np.float32)
    dens = cv2.boxFilter(m, ddepth=-1, ksize=(int(k), int(k)), normalize=True)
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
                "id": int(i),
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

# ----------------------------
# NEW: Weighted density ellipse
# ----------------------------
def fit_density_ellipse(
    dens_u8: np.ndarray,
    *,
    density_thr: int = 120,
    cover_frac: float = 0.90,
    max_points: int = 12000,
):
    """
    Fit an ellipse to maximize captured density mass (approx) using weighted moments.
    - Threshold dens to focus on meaningful regions (still weights by density).
    - Weighted centroid + covariance -> ellipse orientation.
    - Axis lengths chosen to cover `cover_frac` of weighted mass.

    Returns: ellipse (cv2 format) or None
    """
    dens = dens_u8.astype(np.float32)
    m = dens >= float(density_thr)
    if not np.any(m):
        return None

    w = dens * m  # weights
    ys, xs = np.nonzero(m)
    if xs.size < 10:
        return None

    ww = w[ys, xs]

    # Optional subsample for speed if very dense
    if xs.size > max_points:
        # sample proportional to weight (keeps bright areas)
        p = ww / (np.sum(ww) + 1e-9)
        idx = np.random.choice(xs.size, size=max_points, replace=False, p=p)
        xs = xs[idx]
        ys = ys[idx]
        ww = ww[idx]

    W = float(np.sum(ww))
    if W <= 1e-6:
        return None

    # Weighted centroid
    mx = float(np.sum(ww * xs) / W)
    my = float(np.sum(ww * ys) / W)

    dx = xs.astype(np.float32) - mx
    dy = ys.astype(np.float32) - my

    # Weighted covariance (2x2)
    cxx = float(np.sum(ww * dx * dx) / W)
    cyy = float(np.sum(ww * dy * dy) / W)
    cxy = float(np.sum(ww * dx * dy) / W)

    cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float32)

    # Eigen decomposition -> principal axes
    evals, evecs = np.linalg.eigh(cov)
    # sort descending (major axis first)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # angle for OpenCV ellipse (degrees), major axis vector
    vx, vy = float(evecs[0, 0]), float(evecs[1, 0])
    angle = float(np.degrees(np.arctan2(vy, vx)))

    # Project points onto principal axes
    ax0 = evecs[:, 0]  # major
    ax1 = evecs[:, 1]  # minor
    u = dx * ax0[0] + dy * ax0[1]
    v = dx * ax1[0] + dy * ax1[1]

    # Pick radii so ellipse covers `cover_frac` of weighted mass:
    # use weighted quantile of |u| and |v| independently (fast + stable)
    def weighted_quantile_abs(vals, weights, q):
        vals = np.abs(vals.astype(np.float32))
        order = np.argsort(vals)
        vals = vals[order]
        weights = weights[order]
        cdf = np.cumsum(weights) / (np.sum(weights) + 1e-9)
        return float(vals[np.searchsorted(cdf, q, side="left")])

    q = float(np.clip(cover_frac, 0.50, 0.995))
    ru = weighted_quantile_abs(u, ww, q)
    rv = weighted_quantile_abs(v, ww, q)

    # Convert radii to OpenCV ellipse full axis lengths (width,height)
    # Add a small safety pad so it doesn't under-fit
    pad = 1.10
    MA = max(2.0, 2.0 * ru * pad)  # major axis length
    ma = max(2.0, 2.0 * rv * pad)  # minor axis length

    return ((mx, my), (MA, ma), angle)

def refine_best_with_density(
    mask01: np.ndarray,
    *,
    k: int = 31,
    density_thr: int = 120,
    min_area: int = 300,
    max_area: int = 250000,
    ellipse_mode: str = "component",   # "component" or "weighted"
    cover_frac: float = 0.90,          # used by weighted mode
):
    """
    Returns (best, ellipse, dens_u8, dens_mask01).
    dens_mask01 is the thresholded high-density binary mask used for picking.
    """
    dens = mask_density(mask01, k=int(k))
    dens_mask01 = (dens >= int(density_thr)).astype(np.uint8)

    # Always pick a "best" component for bbox/center (fast + consistent with your pipeline)
    best, labels = pick_component_from_mask01(dens_mask01, min_area=min_area, max_area=max_area)

    ellipse = None
    if ellipse_mode == "weighted":
        # Ellipse over the *largest density mass* (not just one solid blob)
        ellipse = fit_density_ellipse(dens, density_thr=int(density_thr), cover_frac=float(cover_frac))
        # If weighted failed but best exists, fallback to component ellipse
        if ellipse is None and best is not None:
            ellipse = fit_ellipse_from_labels(labels, best["id"])
    else:
        # Old behavior: ellipse around the chosen component only
        if best is not None:
            ellipse = fit_ellipse_from_labels(labels, best["id"])

    return best, ellipse, dens, dens_mask01
