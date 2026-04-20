"""Verify that segmentation_data/ uses Pascal VOC mask format.

Checks one image/mask pair:
  1. Mask PNG is mode 'P' (palette / indexed) and same size as the image.
  2. Pixel values are small ints (0..20) plus optional 255 (boundary).
  3. Embedded palette matches the canonical VOC color palette.
"""

from pathlib import Path
import numpy as np
from PIL import Image


VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]


def voc_palette() -> list[int]:
    """Canonical VOC color palette: flat list [r0,g0,b0, r1,g1,b1, ...] for 256 entries."""
    palette = [0] * (256 * 3)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palette[i * 3 : i * 3 + 3] = [r, g, b]
    return palette


def verify_pair(jpg_path: Path, png_path: Path) -> None:
    print(f"image: {jpg_path.name}")
    print(f"mask : {png_path.name}")
    print("-" * 50)

    img = Image.open(jpg_path)
    mask = Image.open(png_path)

    print(f"image  mode={img.mode}  size={img.size}")
    print(f"mask   mode={mask.mode}  size={mask.size}")

    assert img.size == mask.size, "image and mask sizes differ!"
    assert mask.mode == "P", f"expected mode 'P' (paletted), got '{mask.mode}'"

    arr = np.array(mask)
    print(f"mask   shape={arr.shape}  dtype={arr.dtype}")

    uniq = np.unique(arr).tolist()
    print(f"unique pixel values: {uniq}")

    classes_present = [(v, VOC_CLASSES[v] if v < len(VOC_CLASSES) else "boundary/ignore") for v in uniq]
    print("classes:")
    for v, name in classes_present:
        print(f"   {v:3d} -> {name}")

    pal = mask.getpalette()
    assert pal is not None, "mask has no palette (not mode 'P'?)"
    voc = voc_palette()
    n_check = 21 * 3
    matches = pal[:n_check] == voc[:n_check]
    print(f"\nfirst 21 palette entries match VOC palette: {matches}")
    if not matches:
        print("file palette (first 4 RGB):", pal[:12])
        print("VOC  palette (first 4 RGB):", voc[:12])

    voc_ok = (
        mask.mode == "P"
        and arr.shape == img.size[::-1]
        and all((0 <= v <= 20) or v == 255 for v in uniq)
        and matches
    )
    print("\nverdict:", "VOC format" if voc_ok else "NOT VOC format")


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "segmentation_data"
    jpg = data_dir / "2011_002515.jpg"
    png = data_dir / "2011_002515.png"
    verify_pair(jpg, png)
