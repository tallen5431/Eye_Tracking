# eye_ops/components.py
import numpy as np
import cv2
from . import settings as S

def pick_component(mask01):
    num, labels, stats_, centroids = cv2.connectedComponentsWithStats(
        (mask01 * 255).astype(np.uint8), connectivity=8
    )
    h, w = mask01.shape[:2]
    cx0, cy0 = w/2.0, h/2.0

    best = None
    best_score = -1e18

    for i in range(1, num):
        x, y, ww, hh, area = stats_[i]
        if area < S.MIN_AREA or area > S.MAX_AREA:
            continue
        cx, cy = centroids[i]
        aspect = max(ww/max(hh, 1), hh/max(ww, 1))
        dist = np.hypot(cx - cx0, cy - cy0)
        score = area - 40*dist - 400*(aspect - 1.0)
        if score > best_score:
            best_score = score
            best = {
                "id": i,
                "bbox": (x, y, ww, hh),
                "area": int(area),
                "center": (float(cx), float(cy))
            }
    return best, labels
