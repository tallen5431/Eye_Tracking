#!/usr/bin/env python3
"""
eye_animation_clean.py (styled)

Same lightweight share-file-driven eye animation as before, but with the
pixel-matrix / neon-dot visual style from your original `eye_tracking_animation.py`.

Reads pupil center from a JSON share file and moves a stylized eye accordingly.

JSON inputs accepted (flexible):
  1) {"ellipse": {"cx": <float>, "cy": <float>}, "frame_w": <int>, "frame_h": <int>, "confidence": <float>}
  2) {"pupil": {"cx": <float>, "cy": <float>}, "frame_w": <int>, "frame_h": <int>, "confidence": <float>}
  3) {"cx": <float>, "cy": <float>, "frame_w": <int>, "frame_h": <int>, "confidence": <float>}

Optional fields (added by live_tracker_2pass.py):
  - "stage": "FINE" | "COARSE" | "NONE"
  - "timing_ms": float (end-to-end tracking time per frame)
  - "fine_skipped": 0/1 (if fine pass was skipped)

Env vars:
  TRACK_SHARE_FILE: path to JSON (default: ./logs/pupil_share.json)
  TRACK_SENSITIVITY_X / TRACK_SENSITIVITY_Y: motion gain (default: 2.8 / 2.4)
  SMOOTHING: 0..1 exponential smoothing factor (default: 0.18) smaller = smoother/slower
  MAX_OFFSET_FRAC: max excursion as fraction of eye radius (default: 0.55)
  DEADZONE: ignore tiny jitters in normalized units (default: 0.02)
  CALIBRATE_ON_START: 1 to lock initial center as neutral (default: 1)
  FPS: UI update target (default: 60)

Window:
  WIN_W / WIN_H: window size (default 420x420)
  BORDERLESS: 1 for borderless window (default 0)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import tkinter as tk


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class TrackerSample:
    cx: float
    cy: float
    frame_w: Optional[int] = None
    frame_h: Optional[int] = None
    confidence: float = 0.0
    ts: float = 0.0
    stage: str = ""          # "FINE" / "COARSE" / "NONE" (optional)
    timing_ms: float = 0.0   # end-to-end tracker ms (optional)
    fine_skipped: float = 0.0  # 0/1 (optional)


class SharedFileTracker:
    def __init__(self, share_path: Path):
        self.share_path = share_path
        self._last_mtime: float = 0.0
        self._last: Optional[TrackerSample] = None

    def read(self) -> Optional[TrackerSample]:
        # Only reload when the file changes (cheap + avoids jitter)
        try:
            st = self.share_path.stat()
            if st.st_mtime <= self._last_mtime:
                return self._last
            self._last_mtime = st.st_mtime
        except FileNotFoundError:
            return self._last
        except Exception:
            return self._last

        try:
            data = json.loads(self.share_path.read_text(encoding="utf-8"))
        except Exception:
            return self._last

        def pick_center(d):
            if isinstance(d, dict):
                # common nested shapes
                for key in ("ellipse", "pupil", "best", "track", "pos"):
                    v = d.get(key)
                    if isinstance(v, dict) and ("cx" in v and "cy" in v):
                        try:
                            return float(v["cx"]), float(v["cy"])
                        except Exception:
                            pass
                # flat shape
                if "cx" in d and "cy" in d:
                    try:
                        return float(d["cx"]), float(d["cy"])
                    except Exception:
                        pass
            return None

        cc = pick_center(data)
        if not cc:
            return self._last

        cx, cy = cc
        fw = data.get("frame_w") or data.get("w") or data.get("width")
        fh = data.get("frame_h") or data.get("h") or data.get("height")
        conf = float(data.get("confidence", data.get("conf", 0.0)) or 0.0)
        ts = float(data.get("ts", time.time()) or time.time())

        # optional extras (safe defaults)
        stage = str(data.get("stage", "") or "")
        timing_ms = float(data.get("timing_ms", 0.0) or 0.0)
        fine_skipped = float(data.get("fine_skipped", 0.0) or 0.0)

        samp = TrackerSample(
            cx=cx,
            cy=cy,
            frame_w=int(fw) if fw is not None else None,
            frame_h=int(fh) if fh is not None else None,
            confidence=conf,
            ts=ts,
            stage=stage,
            timing_ms=timing_ms,
            fine_skipped=fine_skipped,
        )
        self._last = samp
        return samp


class EyeAnimApp:
    def __init__(self, tracker: SharedFileTracker):
        # Motion controls
        self.sens_x = float(os.environ.get("TRACK_SENSITIVITY_X", "2.8"))
        self.sens_y = float(os.environ.get("TRACK_SENSITIVITY_Y", "2.4"))
        self.smoothing = float(os.environ.get("SMOOTHING", "0.18"))
        self.max_offset_frac = float(os.environ.get("MAX_OFFSET_FRAC", "0.55"))
        self.deadzone = float(os.environ.get("DEADZONE", "0.02"))
        self.calibrate_on_start = os.environ.get("CALIBRATE_ON_START", "1").strip() not in ("0", "false", "False")
        self.fps = int(os.environ.get("FPS", "60"))

        # Window controls
        self.win_w = int(os.environ.get("WIN_W", "420"))
        self.win_h = int(os.environ.get("WIN_H", "420"))
        self.borderless = os.environ.get("BORDERLESS", "0").strip() in ("1", "true", "True")

        # Visual style
        self.BG = "black"
        self.SCLERA_COLOR = "#050505"
        self.DOT_COLOR = "#00f6ff"
        self.DOT_R = 18
        self.SPARK_R = 6

        # Pixel matrix settings
        self.CELL = 8
        self.GAP = 2
        self.BASE_PIXEL = "#0a0a0a"
        self.MID_PIXEL = "#163033"
        self.HOT_PIXEL = self.DOT_COLOR

        self.tracker = tracker

        # Calibration / state
        self.neutral_cx: Optional[float] = None
        self.neutral_cy: Optional[float] = None
        self.neutral_w: Optional[int] = None
        self.neutral_h: Optional[int] = None
        self.dx = 0.0
        self.dy = 0.0

        # last sample (for stage/timing display)
        self._last_sample: Optional[TrackerSample] = None

        # Tk
        self.root = tk.Tk()
        self.root.title("Eye Animation (Styled)")
        self.root.geometry(f"{self.win_w}x{self.win_h}")
        self.root.configure(bg=self.BG)
        if self.borderless:
            self.root.overrideredirect(True)

        self.canvas = tk.Canvas(self.root, width=self.win_w, height=self.win_h, bg=self.BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.info = tk.Label(self.root, text="", fg="#dddddd", bg=self.BG, font=("Consolas", 10), justify="left")
        self.info.place(x=8, y=8)

        # Build graphics
        self._pixels = []  # list: [item_id, pcx, pcy, zone]
        self._draw_static()

        # Keybinds
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("c", lambda e: self._clear_calibration())
        self.root.bind("<Configure>", lambda e: self._on_resize())

        self._schedule()

    def _on_resize(self):
        # Rebuild geometry when window is resized
        self._draw_static()

    def _draw_static(self):
        self.canvas.delete("all")
        self._pixels.clear()

        w = max(10, int(self.canvas.winfo_width() or self.win_w))
        h = max(10, int(self.canvas.winfo_height() or self.win_h))

        self.cx0 = w / 2.0
        self.cy0 = h / 2.0

        # Eye radii
        self.SCLERA_R = min(260, (min(w, h) // 2) - 6)
        self.MATRIX_R = max(10, min(235, self.SCLERA_R - 18))
        self.r_pupil = max(8.0, float(self.DOT_R))

        # Precompute glow radii (squared distances)
        self._HOT_DIST_SQ = float((self.DOT_R * 2.6) ** 2)
        self._MID_DIST_SQ = float((self.DOT_R * 4.2) ** 2)

        # Dark outer circle (sclera)
        self.canvas.create_oval(
            self.cx0 - self.SCLERA_R, self.cy0 - self.SCLERA_R,
            self.cx0 + self.SCLERA_R, self.cy0 + self.SCLERA_R,
            fill=self.SCLERA_COLOR, outline=""
        )

        # Pixel matrix inside sclera
        step = self.CELL + self.GAP
        x0 = self.cx0 - self.MATRIX_R
        y0 = self.cy0 - self.MATRIX_R

        max_gx = int((2 * self.MATRIX_R) // step) + 3
        max_gy = int((2 * self.MATRIX_R) // step) + 3
        r2 = float((self.MATRIX_R - 2) ** 2)

        for gy in range(max_gy):
            for gx in range(max_gx):
                px = x0 + gx * step
                py = y0 + gy * step
                pcx = px + self.CELL / 2
                pcy = py + self.CELL / 2
                if (pcx - self.cx0) ** 2 + (pcy - self.cy0) ** 2 <= r2:
                    item = self.canvas.create_rectangle(
                        px, py, px + self.CELL, py + self.CELL,
                        outline="", fill=self.BASE_PIXEL
                    )
                    self._pixels.append([item, float(pcx), float(pcy), 0])

        # Pupil dot + sparkle
        self.dot_id = self.canvas.create_oval(
            self.cx0 - self.DOT_R, self.cy0 - self.DOT_R,
            self.cx0 + self.DOT_R, self.cy0 + self.DOT_R,
            fill=self.DOT_COLOR, outline=""
        )
        self.spark_id = self.canvas.create_oval(
            self.cx0 - self.SPARK_R, self.cy0 - self.SPARK_R,
            self.cx0 + self.SPARK_R, self.cy0 + self.SPARK_R,
            fill="white", outline=""
        )

    def _clear_calibration(self):
        self.neutral_cx = self.neutral_cy = None
        self.neutral_w = self.neutral_h = None

    def _compute_normalized(self, s: TrackerSample) -> Tuple[float, float, float]:
        # Calibrate: first valid sample becomes neutral center (optional)
        if self.calibrate_on_start and self.neutral_cx is None:
            self.neutral_cx = s.cx
            self.neutral_cy = s.cy
            self.neutral_w = s.frame_w
            self.neutral_h = s.frame_h

        ref_cx = self.neutral_cx if self.neutral_cx is not None else (s.frame_w / 2.0 if s.frame_w else s.cx)
        ref_cy = self.neutral_cy if self.neutral_cy is not None else (s.frame_h / 2.0 if s.frame_h else s.cy)

        fw = s.frame_w or self.neutral_w
        fh = s.frame_h or self.neutral_h
        if fw and fh:
            nx = (s.cx - ref_cx) / (fw / 2.0)
            ny = (s.cy - ref_cy) / (fh / 2.0)
        else:
            nx = (s.cx - ref_cx) / 100.0
            ny = (s.cy - ref_cy) / 100.0

        if abs(nx) < self.deadzone:
            nx = 0.0
        if abs(ny) < self.deadzone:
            ny = 0.0

        nx = clamp(nx, -1.0, 1.0)
        ny = clamp(ny, -1.0, 1.0)

        # NOTE: confidence coming from your tracker is "area" right now (not 0..1).
        # We still clamp, but keep it informative for smoothing:
        conf = float(s.confidence or 0.0)
        conf01 = clamp(conf, 0.0, 1.0) if conf <= 1.5 else 1.0
        return nx, ny, conf01

    def _update_pixels(self, px: float, py: float):
        # Update pixel matrix: only itemconfig when zone changes (fast)
        hot_sq = self._HOT_DIST_SQ
        mid_sq = self._MID_DIST_SQ
        hot_color = self.HOT_PIXEL
        mid_color = self.MID_PIXEL
        base_color = self.BASE_PIXEL
        itemconfig = self.canvas.itemconfig

        for pix in self._pixels:
            ddx = pix[1] - px
            ddy = pix[2] - py
            d_sq = ddx * ddx + ddy * ddy

            if d_sq <= hot_sq:
                new_zone = 2
            elif d_sq <= mid_sq:
                new_zone = 1
            else:
                new_zone = 0

            if new_zone != pix[3]:
                pix[3] = new_zone
                if new_zone == 2:
                    itemconfig(pix[0], fill=hot_color)
                elif new_zone == 1:
                    itemconfig(pix[0], fill=mid_color)
                else:
                    itemconfig(pix[0], fill=base_color)

    def _update_eye(self, nx: float, ny: float, conf01: float):
        max_px = self.SCLERA_R * self.max_offset_frac

        tx = clamp(nx * self.sens_x, -1.0, 1.0) * max_px
        ty = clamp(ny * self.sens_y, -1.0, 1.0) * max_px

        a = clamp(self.smoothing * (0.35 + 0.65 * conf01), 0.02, 0.45)
        self.dx = (1 - a) * self.dx + a * tx
        self.dy = (1 - a) * self.dy + a * ty

        px = self.cx0 + self.dx
        py = self.cy0 + self.dy

        # Clamp pupil inside sclera a bit
        lim = self.SCLERA_R - (self.DOT_R + 10)
        if lim > 10:
            dist = (self.dx * self.dx + self.dy * self.dy) ** 0.5
            if dist > lim:
                s = lim / max(1e-6, dist)
                px = self.cx0 + self.dx * s
                py = self.cy0 + self.dy * s

        # Move dot + sparkle
        self.canvas.coords(
            self.dot_id,
            px - self.DOT_R, py - self.DOT_R,
            px + self.DOT_R, py + self.DOT_R
        )
        self.canvas.coords(
            self.spark_id,
            px - self.SPARK_R + 7, py - self.SPARK_R - 7,
            px + self.SPARK_R + 7, py + self.SPARK_R - 7
        )

        # Glow pixels around dot
        self._update_pixels(px, py)

        # Info text (includes stage + timing)
        stage = (self._last_sample.stage if self._last_sample else "") or "??"
        tms = float(self._last_sample.timing_ms) if self._last_sample else 0.0
        fine_skipped = float(self._last_sample.fine_skipped) if self._last_sample else 0.0
        skipped_txt = " (fine skipped)" if fine_skipped > 0.5 else ""

        self.info.config(
            text=(
                f"nx={nx:+.2f} ny={ny:+.2f}  |  sens=({self.sens_x:.2f},{self.sens_y:.2f})  smooth={self.smoothing:.2f}\n"
                f"stage={stage}{skipped_txt}  timing={tms:.1f}ms  share={self.tracker.share_path}\n"
                f"(ESC quit, 'c' recal)"
            )
        )

    def _tick(self):
        s = self.tracker.read()
        if s is not None:
            self._last_sample = s
            nx, ny, conf01 = self._compute_normalized(s)
            self._update_eye(nx, ny, conf01)

        delay_ms = int(1000 / max(10, self.fps))
        self.root.after(delay_ms, self._tick)

    def _schedule(self):
        self.root.after(30, self._tick)

    def run(self):
        self.root.mainloop()


def main():
    share = os.environ.get("TRACK_SHARE_FILE", "")
    if not share:
        share = str(Path.cwd() / "logs" / "pupil_share.json")
    tracker = SharedFileTracker(Path(share))
    app = EyeAnimApp(tracker)
    app.run()


if __name__ == "__main__":
    main()
