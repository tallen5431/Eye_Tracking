# eye_ops/crop_map.py

def map_point_roi_small_to_rotated(x, y, roi_origin, scale):
    rx, ry = roi_origin
    X = int(round(x / scale)) + int(rx)
    Y = int(round(y / scale)) + int(ry)
    return X, Y

def bbox_roi_small_to_rotated_bbox(bbox_xywh, roi_origin, scale, full_shape_hw):
    x, y, w, h = bbox_xywh
    H, W = full_shape_hw

    X0, Y0 = map_point_roi_small_to_rotated(x, y, roi_origin, scale)
    X1, Y1 = map_point_roi_small_to_rotated(x + w, y + h, roi_origin, scale)

    X0 = max(0, min(W, X0))
    Y0 = max(0, min(H, Y0))
    X1 = max(0, min(W, X1))
    Y1 = max(0, min(H, Y1))

    if X1 <= X0 or Y1 <= Y0:
        return None

    return (X0, Y0, X1, Y1)

def crop_rotated_by_green_bbox(rotated_img_bgr, res):
    best = res.get("best", None)
    if best is None:
        return None, None

    bbox_small = best["bbox"]
    roi_origin = res["roi_origin"]
    scale = float(res["scale"])

    H, W = rotated_img_bgr.shape[:2]
    bb = bbox_roi_small_to_rotated_bbox(bbox_small, roi_origin, scale, (H, W))
    if bb is None:
        return None, None

    X0, Y0, X1, Y1 = bb
    crop = rotated_img_bgr[Y0:Y1, X0:X1].copy()
    return crop, bb

def apply_padding(bb, full_shape, pad_px=0, pad_rel=0.0):
    if bb is None:
        return None

    X0, Y0, X1, Y1 = bb
    H, W = full_shape[:2]

    w = X1 - X0
    h = Y1 - Y0

    rel_pad_x = int(w * pad_rel)
    rel_pad_y = int(h * pad_rel)

    pad_x = pad_px + rel_pad_x
    pad_y = pad_px + rel_pad_y

    X0p = max(0, X0 - pad_x)
    Y0p = max(0, Y0 - pad_y)
    X1p = min(W, X1 + pad_x)
    Y1p = min(H, Y1 + pad_y)

    if X1p <= X0p or Y1p <= Y0p:
        return None

    return (X0p, Y0p, X1p, Y1p)
