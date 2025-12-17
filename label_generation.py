import numpy as np
from pathlib import Path
from tifffile import imread as tiff_imread, imwrite as tiff_imwrite
from PIL import Image

input_path = r""
out_vis_tiff  = r""
out_label_tiff= r""

class_names = [
    "Background",
    "Quartz",
    "Feldspar",
    "Pyrite",
    "Clays",
]


hex_colors = [
    "0",       # Background (black)
    "ffff00",  # Quartz (yellow)
    "ff0000",  # Feldspar (red)
    "00ff00",  # Pyrite (light green)
    "c6c6c6",  # Clays (gray)
]

def hex_to_rgb(h: str):
    """Convert hex string to (R,G,B). Accepts '0', '000', '000000', with/without '#'."""
    h = h.strip().lstrip('#').lower()
    if h in ("0", "00", "000", "0000", "00000"):
        h = "000000"
    if len(h) == 3:
        h = ''.join([c*2 for c in h])
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {h!r}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def read_image_any(path):
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in ('.tif', '.tiff'):
        img = tiff_imread(str(p))
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] >= 3:
            img = img[..., :3]
        else:
            raise ValueError(f"Unexpected TIFF shape: {img.shape}")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    else:
        with Image.open(str(p)) as im:
            im = im.convert("RGB")
            return np.array(im, dtype=np.uint8)

def main():
    print("Reading input:", input_path)
    arr = read_image_any(input_path)
    H, W = arr.shape[:2]
    print("Image size:", W, "x", H, "dtype:", arr.dtype)

    palette = np.array([hex_to_rgb(h) for h in hex_colors], dtype=np.int32)
    flat = arr.reshape(-1, 3).astype(np.int32)

    d2 = np.sum((flat[:, None, :] - palette[None, :, :])**2, axis=2)
    idx = np.argmin(d2, axis=1).astype(np.uint8)

    recolored = palette[idx].reshape(H, W, 3).astype(np.uint8)
    labels = idx.reshape(H, W).astype(np.uint8)

    print("Saving recolored visualization (uncompressed TIFF):", out_vis_tiff)
    tiff_imwrite(out_vis_tiff, recolored, photometric='rgb', compression='none')

    print("Saving label map (uint8 TIFF):", out_label_tiff)
    tiff_imwrite(out_label_tiff, labels, photometric='minisblack', compression='none')

    print("\nClass index mapping:")
    for i, (name, hx) in enumerate(zip(class_names, hex_colors)):
        print(f"  {i}: {name:10s}  #{hx if hx.startswith('#') else '#'+hx}")

    print("Done.")

if __name__ == "__main__":
    main()
