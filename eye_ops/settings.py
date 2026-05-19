# eye_ops/settings.py
"""
Eye tracking pipeline settings
Organized in execution order (2-pass: COARSE -> CROP -> FINE)

Tip:
- For production (no debug drawing), set DRAW_VIZ_COARSE/FINE = False.
- Tune MIN/MAX area separately for coarse vs fine if you use the 2-pass crop.
"""

# ============================================================
# 0) Notebook / batch controls
# ============================================================
MAX_SHOW = 24


# ============================================================
# 1) PASS A (COARSE) — ROI selection on FULL ROTATED image
# ============================================================
# If ROI_FIXED is False, the full image is used as ROI.
ROI_FIXED = False
# Fractional ROI in full rotated image: (x0f, y0f, x1f, y1f)
ROI_FRAC = (0.0, 0.0, 1.0, 1.0)


# ============================================================
# 2) PASS A (COARSE) — Downscale
# ============================================================
# Downscale factor applied to the coarse ROI before processing.
# (Lower = faster, but less detail)
DOWNSCALE = 0.1


# ============================================================
# 3) PASS A/B — Step2 Intensity / Preprocess Mode
# ============================================================
# How to derive the intensity image from ROI BGR:
# "gray", "hsv_v", "lab_l", "min", "g-r",
# "clahe_gray", "clahe_hsv_v", "blackhat_gray", "median_gray"
STEP2_MODE = "clahe_gray"

# CLAHE params (only used in clahe_* modes)
# CLAHE_GAMMA < 1.0 brightens dim images before contrast enhancement (1.0 = off)
CLAHE_GAMMA = 0.6
CLAHE_CLIP = 4.0
CLAHE_TILE = (4, 4)

# Blackhat params (only used in blackhat_gray)
BLACKHAT_K = 21

# Median blur params (only used in median_gray)
MEDIAN_K = 2


# ============================================================
# 3b) PASS A (COARSE) — Bright-guided ROI
# ============================================================
# When enabled, the coarse pass first finds a bright region (sclera /
# surrounding eye tissue) and uses its bounding box to constrain where
# the dark pupil blob is searched.  Pixels outside the bright region are
# masked out before the dark percentile step, so distant dark distractors
# (eyelashes, shadows, clothing) cannot be picked as the pupil.
#
# Falls back to unconstrained dark mask if no bright blob is found.
USE_BRIGHT_COARSE = False

BRIGHT_COARSE_PCT = 85
BRIGHT_COARSE_BLUR_K = 5
BRIGHT_COARSE_OPEN_K = 0
BRIGHT_COARSE_CLOSE_K = 15
BRIGHT_COARSE_DILATE_K = 5
BRIGHT_COARSE_MIN_AREA = 0
BRIGHT_COARSE_MAX_AREA = 250000


# ============================================================
# 4) PASS A/B — Thresholding + Morph Cleanup
# ============================================================
# Percentile threshold: pixels darker than this percentile become mask=1
PCT = 1

BLUR_K = 3

OPEN_K = 0
CLOSE_K = 35

# Hole fill (fix glare "donuts")
FILL_HOLES = False
FILL_HOLES_FINE = False

# Density-refine (optional): fit overlays using high-density regions of the mask
USE_DENSITY_REFINE_FINE = True
DENSITY_K_FINE = 35
DENSITY_THR_FINE = 40

# Density-refine ellipse behavior (fine)
DENSITY_ELLIPSE_MODE_FINE = "weighted"
DENSITY_COVER_FRAC_FINE = 0.50


# ============================================================
# 5) PASS A/B — Blob selection filters
# ============================================================
MIN_AREA = 0
MAX_AREA = 25000


# ============================================================
# 6) PASS A/B — Optional Adaptive threshold (alternative to percentile)
# ============================================================
USE_ADAPTIVE = False
ADAPT_BLOCK = 31
ADAPT_C = 7


# ============================================================
# 7) PASS A/B — Ellipse fitting
# ============================================================
DO_ELLIPSE = True


# ============================================================
# 8) CROP STEP — From coarse bbox → padded crop on FULL ROTATED
# ============================================================
PAD_PX = 100
PAD_REL = 0.0


# ============================================================
# 9) PASS B (FINE) — Run pipeline on the padded crop
# ============================================================
DOWNSCALE_FINE = 0.2

PCT_FINE = 5
BLUR_K_FINE = 0
OPEN_K_FINE = 0
CLOSE_K_FINE = 11

MIN_AREA_FINE = 0
MAX_AREA_FINE = 250000


# ============================================================
# 9b) PASS B (FINE) — Glare / bright-pixel mask
# ============================================================
USE_GLARE_MASK_FINE = False

GLARE_PCT_FINE = 99
GLARE_BLUR_K_FINE = 3
GLARE_OPEN_K_FINE = 3
GLARE_CLOSE_K_FINE = 11
GLARE_DILATE_K_FINE = 0


# ============================================================
# 10) DEV / DEBUG — overlay drawing controls (timing + visualization)
# ============================================================
DRAW_VIZ_COARSE = True
DRAW_VIZ_FINE = True
