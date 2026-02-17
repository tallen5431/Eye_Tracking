# eye_ops/density_refine.py
import numpy as np
import cv2


def _odd(k: int) -> int:
    k = int(k)
    return k if (k % 2 == 1) else (k + 1)


def mask_density(mask01: np.ndarray, k: int = 31) -> np.ndarray:
    """
    Local dark-pixel density from a 0/1 mask.
    Returns uint8 0..255.
    """
    k = _odd(k)
    m = mask01.astype(np.float32)
    dens = cv2.boxFilter(m, ddepth=-1, ksize=(k, k), normalize=True)
    return (dens * 255.0).astype(np.uint8)


def _mask01_to_255(mask01: np.ndarray) -> np.ndarray:
    return (mask01.astype(np.uint8) * 255)


def _mask255_to_01(mask255: np.ndarray) -> np.ndarray:
    return (mask255 > 0).astype(np.uint8)


def refine_best_with_density(
    mask01: np.ndarray,
    *,
    k: int = 31,
    density_thr: int = 120,
    min_area: int = 300,
    max_area: int = 250000,
    close_k: int = 0,
):
    """
    FAST density refine (union-fit):
      - builds density map on mask01
      - thresholds to dens_mask
      - optional morphological CLOSE to bridge glare gaps
      - computes bbox + center from ALL dense pixels (no components)
      - fits ellipse on ALL dense pixels

    Returns: (best, ellipse, dens_u8, dens_mask01)

    best is:
      {"id": 1, "bbox": (x,y,w,h), "area": area_px, "center": (cx,cy)}
    ellipse is cv2.fitEllipse format or None
    dens_mask01 is 0/1 mask (thresholded density, after optional close)
    """
    dens_u8 = mask_density(mask01, k=k)

    # Threshold to binary (255) for OpenCV routines
    dens_mask255 = np.where(dens_u8 >= int(density_thr), 255, 0).astype(np.uint8)

    # Optional: CLOSE to connect islands / seal glare gaps
    if int(close_k) > 0:
        kk = int(close_k)
        if kk % 2 == 0:
            kk += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
        dens_mask255 = cv2.morphologyEx(dens_mask255, cv2.MORPH_CLOSE, kernel)

    # Find all non-zero pixels (C-optimized)
    pts = cv2.findNonZero(dens_mask255)
    if pts is None:
        return None, None, dens_u8, _mask255_to_01(dens_mask255)

    area = int(len(pts))
    if area < int(min_area) or area > int(max_area):
        # Still return density outputs for debug, but no detection
        return None, None, dens_u8, _mask255_to_01(dens_mask255)

    # BBox from all points (C-optimized)
    x, y, w, h = cv2.boundingRect(pts)

    # Center from image moments (C-optimized)
    M = cv2.moments(dens_mask255, binaryImage=True)
    if M.get("m00", 0.0) > 0.0:
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
    else:
        # Fallback: bbox center
        cx = float(x + w * 0.5)
        cy = float(y + h * 0.5)

    best = {
        "id": 1,
        "bbox": (int(x), int(y), int(w), int(h)),
        "area": int(area),
        "center": (cx, cy),
    }

    # Ellipse from ALL points (needs >=5)
    ellipse = None
    if area >= 5:
        try:
            ellipse = cv2.fitEllipse(pts)
        except Exception:
            ellipse = None

    dens_mask01 = _mask255_to_01(dens_mask255)
    return best, ellipse, dens_u8, dens_mask01
