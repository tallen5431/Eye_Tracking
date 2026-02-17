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

def mask_percentile(gray, *, pct=None, blur_k=None, open_k=None, close_k=None):
    pct = float(S.PCT if pct is None else pct)
    blur_k = _odd(S.BLUR_K if blur_k is None else blur_k)
    open_k = int(S.OPEN_K if open_k is None else open_k)
    close_k = int(S.CLOSE_K if close_k is None else close_k)

    g = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    thr = float(np.percentile(g, pct))
    m = (g < thr).astype(np.uint8)
    return thr, morph(m, open_k=open_k, close_k=close_k)
