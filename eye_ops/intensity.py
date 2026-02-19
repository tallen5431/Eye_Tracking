# eye_ops/intensity.py
import numpy as np
import cv2
from . import settings as S

def step2_intensity(roi_small_bgr):
    mode = S.STEP2_MODE

    if mode == "gray":
        return cv2.cvtColor(roi_small_bgr, cv2.COLOR_BGR2GRAY)

    if mode == "hsv_v":
        hsv = cv2.cvtColor(roi_small_bgr, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 2]

    if mode == "lab_l":
        lab = cv2.cvtColor(roi_small_bgr, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0]

    if mode == "min":
        return np.min(roi_small_bgr, axis=2).astype(np.uint8)

    if mode == "g-r":
        b, g, r = cv2.split(roi_small_bgr)
        return cv2.subtract(g, r)

    if mode == "clahe_gray":
        g = cv2.cvtColor(roi_small_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=float(S.CLAHE_CLIP), tileGridSize=tuple(S.CLAHE_TILE))
        return clahe.apply(g)

    if mode == "clahe_hsv_v":
        hsv = cv2.cvtColor(roi_small_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        clahe = cv2.createCLAHE(clipLimit=float(S.CLAHE_CLIP), tileGridSize=tuple(S.CLAHE_TILE))
        return clahe.apply(v)

    if mode == "blackhat_gray":
        g = cv2.cvtColor(roi_small_bgr, cv2.COLOR_BGR2GRAY)
        k = int(S.BLACKHAT_K)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)

    if mode == "median_gray":
        g = cv2.cvtColor(roi_small_bgr, cv2.COLOR_BGR2GRAY)
        k = int(S.MEDIAN_K)
        if k % 2 == 0:
            k += 1
        return cv2.medianBlur(g, k)

    return cv2.cvtColor(roi_small_bgr, cv2.COLOR_BGR2GRAY)
