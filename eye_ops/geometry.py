# eye_ops/geometry.py
import numpy as np
import cv2

def fit_ellipse(labels, comp_id):
    ys, xs = np.where(labels == comp_id)
    if len(xs) < 20:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
    if len(pts) < 5:
        return None
    try:
        return cv2.fitEllipse(pts)
    except:
        return None
