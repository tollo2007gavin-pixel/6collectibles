"""Image loading that respects EXIF orientation.

cv2.imread ignores the EXIF orientation tag, so an iPhone portrait photo comes
back as a landscape array. That silently breaks aspect-ratio sanity checks and
any assumption about which card edge is 'top'. Always load through here.
"""
import cv2
import numpy as np
from PIL import Image, ExifTags

_ORIENT_TAG = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")


def load_bgr(path):
    """Load an image as BGR uint8 with EXIF orientation applied."""
    pil = Image.open(path)
    exif = pil.getexif()
    orient = exif.get(_ORIENT_TAG, 1) if exif else 1
    rgb = np.array(pil.convert("RGB"))
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if orient == 3:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif orient == 6:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif orient == 8:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def downscale(img, max_dim=1600):
    """Return (small_img, scale) where scale maps small -> full coords."""
    h, w = img.shape[:2]
    s = max(h, w) / float(max_dim)
    if s <= 1.0:
        return img.copy(), 1.0
    small = cv2.resize(img, (int(round(w / s)), int(round(h / s))),
                       interpolation=cv2.INTER_AREA)
    return small, s
