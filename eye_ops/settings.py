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
ROI_FIXED = True
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
STEP2_MODE = "gray"

# CLAHE params (only used in clahe_* modes)
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# Blackhat params (only used in blackhat_gray)
BLACKHAT_K = 21

# Median blur params (only used in median_gray)
MEDIAN_K = 2


# ============================================================
# 4) PASS A/B — Thresholding + Morph Cleanup
# ============================================================
# Percentile threshold: pixels darker than this percentile become mask=1
# PCT = 15 Original
PCT = 1

# Gaussian blur kernel size used before thresholding (odd recommended)
BLUR_K = 3
# BLUR_K = 9
# BLUR_K = 0
# BLUR_K = 17
# BLUR_K = 27

# Morphology kernel sizes (0 disables)

OPEN_K = 0
# OPEN_K = 1
# OPEN_K = 3
# OPEN_K = 5
# OPEN_K = 9


# CLOSE_K = 0
# CLOSE_K = 1
# CLOSE_K = 3
# CLOSE_K = 5
# CLOSE_K = 17
CLOSE_K = 35
# CLOSE_K = 0

# Hole fill (fix glare "donuts")
FILL_HOLES = False

# Optional: separate toggle for fine pass only
# FILL_HOLES_FINE = False
FILL_HOLES_FINE = False

# Density-refine (optional): fit overlays using high-density regions of the mask
USE_DENSITY_REFINE_FINE = True
DENSITY_K_FINE = 35 #Smoothing of the mask, density gradient
DENSITY_THR_FINE = 40   # 0..255 (higher = only very dense regions)

# Density-refine ellipse behavior (fine)
DENSITY_ELLIPSE_MODE_FINE = "weighted"   # "weighted" or "component"
DENSITY_COVER_FRAC_FINE = 0.50           # 0.85..0.95 typical (Total density the ellipse tries to cover)


# ============================================================
# 5) PASS A/B — Blob selection filters
# ============================================================
# Coarse blob area bounds in COARSE roi_small pixels
# MIN_AREA = 300
MIN_AREA = 0
MAX_AREA = 25000


# ============================================================
# 6) PASS A/B — Optional Adaptive threshold (alternative to percentile)
# ============================================================
USE_ADAPTIVE = False
ADAPT_BLOCK = 31   # must be odd; code will fix if even
ADAPT_C = 7


# ============================================================
# 7) PASS A/B — Ellipse fitting
# ============================================================
DO_ELLIPSE = True


# ============================================================
# 8) CROP STEP — From coarse bbox → padded crop on FULL ROTATED
# ============================================================
# Padding around mapped bbox (full-image pixels and relative fraction)
# PAD_PX = 0
PAD_PX = 100

PAD_REL = 0.0
# PAD_REL = 0.5

# ============================================================
# 9) PASS B (FINE) — Run pipeline on the padded crop
# ============================================================
# Downscale factor applied to the *cropped* ROI before processing.
# 1.0 = full-res crop; 0.5 = half-res crop (faster).
# DOWNSCALE_FINE = 0.1
DOWNSCALE_FINE = 0.2

# Fine-pass mask tuning (overrides coarse if set)
# PCT_FINE = 10
PCT_FINE = 5 #Controls how much of the heat map there is by darkest pixels.

# Fine-pass mask tuning

# BLUR_K_FINE = 1         # or 11 if glare is strong
BLUR_K_FINE = 0         # or 11 if glare is strong, Rounds edges of heat map
# BLUR_K_FINE = 35

OPEN_K_FINE = 0
# OPEN_K_FINE = 3  # lower opening so you don't break blobs

# CLOSE_K_FINE = 0
CLOSE_K_FINE = 17        # increase closing to bridge glare gaps, closes holes in heat map
# CLOSE_K_FINE = 27        # increase closing to bridge glare gaps
# CLOSE_K_FINE = 35        # increase closing to bridge glare gaps
# CLOSE_K_FINE = 55
# CLOSE_K_FINE = 75

# Fine-pass blob area bounds in FINE roi_small pixels.
# (Recommended to tune separately; crop size varies.)
# MIN_AREA_FINE = 1000
MIN_AREA_FINE = 0
MAX_AREA_FINE = 250000


# ============================================================
# 9b) PASS B (FINE) — Glare / bright-pixel mask
# ============================================================
# When enabled, bright pixels (specular reflections / glare) are detected
# AFTER the first ROI crop and subtracted from the dark pupil mask so that
# glare does not break the pupil blob apart.
#
# Only applied during the FINE pass (after coarse ROI is known).
USE_GLARE_MASK_FINE = False

# Percentile used to call a pixel "bright" (e.g. 99 → top 1% brightest).
GLARE_PCT_FINE = 99

# Gaussian blur before glare thresholding (odd; 1 disables blur).
GLARE_BLUR_K_FINE = 3

# Morphological opening (removes tiny specks); 0 to disable.
GLARE_OPEN_K_FINE = 3

# Morphological closing (fills glare blob gaps); 0 to disable.
GLARE_CLOSE_K_FINE = 11

# Optional dilation of the glare mask before subtraction (grows exclusion zone).
# 0 to disable.
GLARE_DILATE_K_FINE = 0


# ============================================================
# 10) DEV / DEBUG — overlay drawing controls (timing + visualization)
# ============================================================
# These only affect the "viz" images (debug overlays).
# Set False for production eye tracking to measure compute-only.
DRAW_VIZ_COARSE = True
DRAW_VIZ_FINE = True
