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


def _atomic_write_json(path, payload):
    """
    Best-effort atomic JSON write on Windows.

    - Writes to a unique temp file in the same directory.
    - Tries os.replace() with retries (WinError 5 can happen if target is locked/read-only/AV).
    - Falls back to direct write if replace keeps failing.
    """
    import json, os, time
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = json.dumps(payload, ensure_ascii=False)

    # Unique temp name avoids collisions if multiple writers run
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")

    # Write temp
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

    # Try atomic replace with retries
    for attempt in range(30):
        try:
            os.replace(str(tmp), str(path))
            return
        except PermissionError:
            # Common on Windows when file is briefly locked or marked read-only
            time.sleep(0.005 * (attempt + 1))
        except OSError:
            time.sleep(0.005 * (attempt + 1))

    # Fallback: direct write (non-atomic, but avoids hard crash)
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

def _pick_best_result(res: dict) -> dict | None:
    if not isinstance(res, dict):
        return None
    if res.get("best") is not None:
        return res
    c = res.get("coarse")
    if isinstance(c, dict) and c.get("best") is not None:
        return c
    return None


def _pick_stage_source(res: dict) -> tuple[str, dict | None]:
    """
    Returns (stage_label, src_result_dict_for_overlay_or_mask)
    stage_label: "FINE" / "COARSE" / "NONE"
    """
    if isinstance(res, dict) and res.get("best") is not None:
        return "FINE", res
    c = res.get("coarse") if isinstance(res, dict) else None
    if isinstance(c, dict) and c.get("best") is not None:
        return "COARSE", c
    return "NONE", None


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


def _density_u8_from_mask(mask01, k: int = 31) -> cv2.Mat:
    """
    Very fast local density visualization: box filter over 0/1 mask -> 0..255.
    Only used for preview/debug, so we compute it only when requested.
    """
    k = int(k)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    m = mask01.astype("float32")
    dens = cv2.boxFilter(m, ddepth=-1, ksize=(k, k), normalize=True)
    return (dens * 255.0).astype("uint8")


def main():
    share_path = Path(os.environ.get("TRACK_SHARE_FILE", str(Path.cwd() / "logs" / "pupil_share.json")))
    cam_id = int(os.environ.get("CAMERA_ID", "0"))
    rotate_cw = os.environ.get("ROTATE_90_CW", "1") not in ("0", "false", "False")

    # Target processing cadence (your tracker loop)
    target_fps = float(os.environ.get("TARGET_FPS", os.environ.get("FPS", "60")))
    write_when_missing = os.environ.get("WRITE_WHEN_MISSING", "0") in ("1", "true", "True")
    disable_viz = os.environ.get("DISABLE_VIZ", "1") in ("1", "true", "True")

    # Camera capture hints (optional)
    cam_w = int(os.environ.get("CAM_W", "0"))
    cam_h = int(os.environ.get("CAM_H", "0"))
    cam_fps = float(os.environ.get("CAM_FPS", "0"))
    cam_backend = os.environ.get("CAM_BACKEND", "auto").lower()  # "dshow", "msmf", "auto"

    # Preview controls
    preview = os.environ.get("PREVIEW", "1") in ("1", "true", "True")
    preview_scale = float(os.environ.get("PREVIEW_SCALE", "0.75"))
    window_name = os.environ.get("PREVIEW_WINDOW", "2-pass Tracker Preview")
    density_k = int(os.environ.get("DENSITY_K_PREVIEW", "31"))

    if disable_viz:
        S.DRAW_VIZ_COARSE = False
        S.DRAW_VIZ_FINE = False

    # Open camera
    api = 0
    if cam_backend == "dshow":
        api = cv2.CAP_DSHOW
    elif cam_backend == "msmf":
        api = cv2.CAP_MSMF

    cap = cv2.VideoCapture(cam_id, api) if api != 0 else cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        # fallback
        cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera id {cam_id}")

    # Try set capture props if provided
    if cam_w > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cam_w))
    if cam_h > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cam_h))
    if cam_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(cam_fps))

    last_payload = None
    frame_interval = 1.0 / max(1e-6, target_fps)

    paused = False
    show_overlay = True

    # preview_mode: 0=frame(+overlay), 1=mask, 2=density
    preview_mode = 0

    if preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("===================================================")
    print("Live 2-pass tracker writer (with preview)")
    print(" Camera ID:", cam_id)
    print(" Rotate CW:", rotate_cw)
    print(" Share file:", share_path)
    print(" Target FPS:", target_fps)
    print(" Capture set:", f"{cam_w}x{cam_h}" if (cam_w or cam_h) else "(default)", "fps" if cam_fps else "", cam_fps or "")
    print(" Preview:", preview, f"(scale={preview_scale})")
    print(" Hotkeys: q/ESC=quit, space=pause, o=toggle overlay, m=mask, d=density")
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

            stage_label, src_for_debug = _pick_stage_source(res)
            fine_skipped = float(timing.get("fine_skipped", 0.0) or 0.0)
            ms_total = float(1000.0 * timing.get("total_2pass", timing.get("fine_total", timing.get("total", 0.0))))

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
                        "timing_ms": ms_total,
                        "stage": stage_label,
                        "fine_skipped": fine_skipped,
                    }
                    _atomic_write_json(share_path, payload)
                    last_payload = payload
            else:
                if write_when_missing and last_payload is not None:
                    last_payload["ts"] = float(time.time())
                    last_payload["timing_ms"] = ms_total
                    last_payload["stage"] = stage_label
                    last_payload["fine_skipped"] = fine_skipped
                    _atomic_write_json(share_path, last_payload)

            # Preview frame assembly (ALWAYS define disp when preview)
            if preview:
                disp = frame

                # Decide which mask source to show if we are in mask/density mode
                mask_src = None
                if isinstance(res, dict) and res.get("mask_final") is not None and res.get("best") is not None:
                    mask_src = res  # fine
                elif isinstance(res, dict) and isinstance(res.get("coarse"), dict) and res["coarse"].get("mask_final") is not None:
                    mask_src = res["coarse"]  # coarse

                if preview_mode == 1 and mask_src is not None:
                    mask01 = mask_src["mask_final"]
                    disp = cv2.cvtColor((mask01.astype("uint8") * 255), cv2.COLOR_GRAY2BGR)
                elif preview_mode == 2 and mask_src is not None:
                    mask01 = mask_src["mask_final"]
                    dens = _density_u8_from_mask(mask01, k=density_k)
                    disp = cv2.applyColorMap(dens, cv2.COLORMAP_INFERNO)

                # Overlay only in frame mode (preview_mode==0) and if enabled
                overlay_src_label = "NONE"
                if preview_mode == 0 and show_overlay:
                    src = src_for_debug  # already fine/coarse/None chosen by stage
                    overlay_src_label = stage_label
                    if src is not None:
                        disp = draw_pupil_overlay_on_full(disp, src, draw_bbox=True)

                # Status text (always)
                fps_est = 1.0 / max(1e-9, (time.perf_counter() - t0))
                mode_name = ["frame", "mask", "density"][preview_mode]
                fine_note = " (fine skipped)" if fine_skipped > 0.5 else ""
                txt1 = f"fps~{fps_est:5.1f}  ms={ms_total:5.1f}  stage={stage_label}{fine_note}"
                txt2 = f"mode={mode_name}  overlay={'on' if show_overlay else 'off'}({overlay_src_label})  paused={'yes' if paused else 'no'}"
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
                preview_mode = 1  # mask
            elif k == ord("d"):
                preview_mode = 2  # density
            elif k == ord("f"):
                preview_mode = 0  # back to frame

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
