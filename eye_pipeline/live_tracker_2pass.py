#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import json
from pathlib import Path
import cv2

from eye_ops import settings as S
from eye_pipeline.pipeline_2pass import run_pipeline_on_image
from eye_ops.overlay import draw_pupil_overlay_on_full


def _atomic_write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=0), encoding="utf-8")
    tmp.replace(path)


def _pick_best_result(res: dict) -> dict | None:
    if res is None:
        return None
    if res.get("best") is not None:
        return res
    c = res.get("coarse")
    if isinstance(c, dict) and c.get("best") is not None:
        return c
    return None


def _map_center_to_full(res_used: dict) -> tuple[float, float] | None:
    best = res_used.get("best")
    if best is None:
        return None
    cx, cy = best["center"]
    ox, oy = res_used.get("roi_origin", (0, 0))
    scale = float(res_used.get("scale", 1.0))
    X = (float(cx) / scale) + float(ox)
    Y = (float(cy) / scale) + float(oy)
    return X, Y


def main():
    share_path = Path(os.environ.get("TRACK_SHARE_FILE", str(Path.cwd() / "logs" / "pupil_share.json")))
    cam_id = int(os.environ.get("CAMERA_ID", "0"))
    rotate_cw = os.environ.get("ROTATE_90_CW", "1") not in ("0", "false", "False")
    target_fps = float(os.environ.get("TARGET_FPS", os.environ.get("FPS", "60")))
    write_when_missing = os.environ.get("WRITE_WHEN_MISSING", "0") in ("1", "true", "True")
    disable_viz = os.environ.get("DISABLE_VIZ", "1") in ("1", "true", "True")

    # Preview controls
    preview = os.environ.get("PREVIEW", "1") in ("1", "true", "True")
    preview_scale = float(os.environ.get("PREVIEW_SCALE", "0.75"))  # shrink window a bit
    window_name = os.environ.get("PREVIEW_WINDOW", "2-pass Tracker Preview")

    if disable_viz:
        S.DRAW_VIZ_COARSE = False
        S.DRAW_VIZ_FINE = False

    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera id {cam_id}")

    last_payload = None
    frame_interval = 1.0 / max(1e-6, target_fps)

    paused = False
    show_overlay = True
    show_mask = False

    if preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("===================================================")
    print("Live 2-pass tracker writer (with preview)")
    print(" Camera ID:", cam_id)
    print(" Rotate CW:", rotate_cw)
    print(" Share file:", share_path)
    print(" Target FPS:", target_fps)
    print(" Preview:", preview, f"(scale={preview_scale})")
    print(" Hotkeys: q/ESC=quit, space=pause, o=overlay, m=mask")
    print("===================================================")

    while True:
        t0 = time.perf_counter()

        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            if rotate_cw:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Run pipeline
            res, timing = run_pipeline_on_image(frame)
            used = _pick_best_result(res)

            # Write share file
            if used is not None:
                center = _map_center_to_full(used)
                if center is not None:
                    X, Y = center
                    H, W = frame.shape[:2]
                    payload = {
                        "ellipse": {"cx": float(X), "cy": float(Y)},
                        "frame_w": int(W),
                        "frame_h": int(H),
                        "confidence": float(used.get("best", {}).get("area", 0) or 0.0),
                        "ts": float(time.time()),
                        "timing_ms": float(1000.0 * timing.get("total_2pass", timing.get("fine_total", timing.get("total", 0.0)))),
                    }
                    _atomic_write_json(share_path, payload)
                    last_payload = payload
            else:
                if write_when_missing and last_payload is not None:
                    last_payload["ts"] = float(time.time())
                    _atomic_write_json(share_path, last_payload)

            # Preview frame assembly
            if preview:
                disp = frame
                if show_mask:
                    # show fine mask if available, else coarse mask
                    mask = None
                    if isinstance(res, dict) and res.get("mask_final") is not None:
                        mask = res["mask_final"]
                    if mask is None and isinstance(res.get("coarse"), dict) and res["coarse"].get("mask_final") is not None:
                        mask = res["coarse"]["mask_final"]
                    if mask is not None:
                        disp = cv2.cvtColor((mask.astype("uint8") * 255), cv2.COLOR_GRAY2BGR)

            overlay_src_label = "NONE"
            
            if show_overlay and not show_mask:
                # Prefer fine overlay on full frame, fallback to coarse
                src = None
                if isinstance(res, dict) and res.get("best") is not None:
                    src = res
                    overlay_src_label = "FINE"
                elif isinstance(res, dict) and isinstance(res.get("coarse"), dict) and res["coarse"].get("best") is not None:
                    src = res["coarse"]
                    overlay_src_label = "COARSE"
            
                if src is not None:
                    disp = draw_pupil_overlay_on_full(disp, src, draw_bbox=True)


                # Add status text
                fps_est = 1.0 / max(1e-9, (time.perf_counter() - t0))
                txt1 = f"fps~{fps_est:5.1f}  ms={timing.get('total_2pass', 0)*1000:4.1f}"
                txt2 = (
                    f"overlay={'on' if show_overlay else 'off'}({overlay_src_label})  "
                    f"mask={'on' if show_mask else 'off'}  "
                    f"paused={'yes' if paused else 'no'}"
                )
                cv2.putText(disp, txt1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(disp, txt2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if preview_scale != 1.0:
                    disp = cv2.resize(disp, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)

                cv2.imshow(window_name, disp)

        # Key handling (works even when paused)
        if preview:
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):  # ESC or q
                break
            elif k == ord(" "):
                paused = not paused
            elif k == ord("o"):
                show_overlay = not show_overlay
            elif k == ord("m"):
                show_mask = not show_mask

        # FPS cap (only when not paused)
        if not paused:
            dt = time.perf_counter() - t0
            sleep_for = frame_interval - dt
            if sleep_for > 0:
                time.sleep(sleep_for)

    cap.release()
    if preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
