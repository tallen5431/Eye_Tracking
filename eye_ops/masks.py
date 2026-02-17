# eye_ops/masks.py
import numpy as np
import cv2
from . import settings as S

def _odd(k: int) -> int:
    k = int(k)
    return k if (k % 2 == 1) else (k + 1)

def morph(mask, open_k: int, close_k: int):
    m = mask.copy().astype(np.uint8)
    if int(open_k) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_k), int(open_k)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if int(close_k) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_k), int(close_k)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m

def fill_holes(mask01: np.ndarray) -> np.ndarray:
    """
    Fill interior holes in a binary mask (0/1).
    """
    m = (mask01.astype(np.uint8) * 255)
    h, w = m.shape[:2]
    flood = m.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)   # fill background
    holes = cv2.bitwise_not(flood)                  # holes become white
    filled = cv2.bitwise_or(m, holes)
    return (filled > 0).astype(np.uint8)


def mask_percentile(gray, *, pct=None, blur_k=None, open_k=None, close_k=None, fill=None):
    pct = float(S.PCT if pct is None else pct)
    blur_k = _odd(S.BLUR_K if blur_k is None else blur_k)
    open_k = int(S.OPEN_K if open_k is None else open_k)
    close_k = int(S.CLOSE_K if close_k is None else close_k)

    # If fill not specified, use settings default
    if fill is None:
        fill = bool(getattr(S, "FILL_HOLES", True))

    g = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    thr = float(np.percentile(g, pct))
    m = (g < thr).astype(np.uint8)

    m2 = morph(m, open_k=open_k, close_k=close_k)
    if fill:
        m2 = fill_holes(m2)

    return thr, m2


def mask_adaptive(gray, *, blur_k=None, open_k=None, close_k=None, block=None, C=None):
    """
    Adaptive threshold mask (optional alternative to percentile).
    Returns a cleaned 0/1 mask.
    """
    blur_k = _odd(S.BLUR_K if blur_k is None else blur_k)
    open_k = int(S.OPEN_K if open_k is None else open_k)
    close_k = int(S.CLOSE_K if close_k is None else close_k)

    block = int(S.ADAPT_BLOCK if block is None else block)
    if block % 2 == 0:
        block += 1
    C = int(S.ADAPT_C if C is None else C)

    g = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    m255 = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block, C
    )
    return morph((m255 > 0).astype(np.uint8), open_k=open_k, close_k=close_k)
