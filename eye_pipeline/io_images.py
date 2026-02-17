# eye_pipeline/io_images.py

from pathlib import Path
import cv2

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Default images directory (inside this package)
DEFAULT_IMAGE_DIR = Path(__file__).parent / "images"


def load_rotated_images(img_dir: Path | None = None, max_images: int = 100):
    """
    Loads images, rotates them 90° clockwise.
    
    Parameters:
        img_dir: Path to images folder (optional).
                 If None, uses eye_pipeline/images.
        max_images: Max number of images to load.
    
    Returns:
        images (list of BGR numpy arrays),
        kept_paths (list of Path objects)
    """

    if img_dir is None:
        img_dir = DEFAULT_IMAGE_DIR
    else:
        img_dir = Path(img_dir)

    if not img_dir.exists():
        raise RuntimeError(f"Image directory does not exist: {img_dir}")

    paths = [p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in EXTS]
    paths = paths[:max_images]

    images = []
    kept_paths = []

    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: could not load {p}")
            continue

        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        images.append(img)
        kept_paths.append(p)

    if not images:
        raise RuntimeError(f"No images loaded from {img_dir.resolve()}")

    print(f"Loaded {len(images)} images from {img_dir.resolve()}")
    return images, kept_paths
