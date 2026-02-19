# eye_ops/roi.py
import cv2
from . import settings as S

def crop_roi(img_bgr):
    H, W = img_bgr.shape[:2]
    if not S.ROI_FIXED:
        return img_bgr, (0, 0)
    x0f, y0f, x1f, y1f = S.ROI_FRAC
    x0 = int(W * x0f); y0 = int(H * y0f)
    x1 = int(W * x1f); y1 = int(H * y1f)
    return img_bgr[y0:y1, x0:x1].copy(), (x0, y0)

def downscale(img):
    if float(S.DOWNSCALE) == 1.0:
        return img, 1.0
    out = cv2.resize(img, None, fx=S.DOWNSCALE, fy=S.DOWNSCALE, interpolation=cv2.INTER_AREA)
    return out, float(S.DOWNSCALE)
