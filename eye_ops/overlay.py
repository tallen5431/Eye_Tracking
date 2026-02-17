# eye_ops/overlay.py
import cv2
from .crop_map import map_point_roi_small_to_rotated, bbox_roi_small_to_rotated_bbox

def ellipse_roi_small_to_rotated(ellipse, roi_origin, scale):
    """
    Convert cv2.fitEllipse output from roi_small coords -> full ROTATED coords.
    ellipse: ((cx,cy),(MA,ma),angle)
    """
    if ellipse is None:
        return None
    (cx, cy), (MA, ma), angle = ellipse
    Xc, Yc = map_point_roi_small_to_rotated(cx, cy, roi_origin, scale)
    s = float(scale)
    return ((Xc, Yc), (float(MA) / s, float(ma) / s), float(angle))

def draw_pupil_overlay_on_full(rotated_full_bgr, res, *, draw_bbox=False):
    """
    Draw red dot + cyan ellipse from a result dict (coarse OR fine) onto the FULL image.
    Uses:
      res["best"]["center"], res["best"]["bbox"], res["ellipse"], res["roi_origin"], res["scale"]
    """
    out = rotated_full_bgr.copy()
    best = res.get("best")
    if best is None:
        return out

    roi_origin = res["roi_origin"]
    scale = float(res.get("scale", 1.0))

    # Dot
    cx, cy = best["center"]
    Xc, Yc = map_point_roi_small_to_rotated(cx, cy, roi_origin, scale)
    cv2.circle(out, (int(Xc), int(Yc)), 5, (0, 0, 255), -1)

    # Ellipse
    ell_full = ellipse_roi_small_to_rotated(res.get("ellipse"), roi_origin, scale)
    if ell_full is not None:
        cv2.ellipse(out, ell_full, (255, 255, 0), 2)

    # Optional bbox
    if draw_bbox:
        H, W = out.shape[:2]
        bb = bbox_roi_small_to_rotated_bbox(best["bbox"], roi_origin, scale, (H, W))
        if bb is not None:
            X0, Y0, X1, Y1 = bb
            cv2.rectangle(out, (X0, Y0), (X1, Y1), (0, 255, 0), 2)

    return out
