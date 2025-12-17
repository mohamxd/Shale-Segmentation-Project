import os
import numpy as np
import cv2
from tifffile import imread, imwrite

original_path = r""
cropped_path  = r""
output_path   = r""

def read_image(path):
    """Read TIFF using tifffile (keeps dtype and channels)."""
    img = imread(path)
    if img.ndim == 3 and img.shape[0] <= 4 and img.shape[0] > 1 and img.shape[0] < img.shape[1]:
        if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            img = np.transpose(img, (1, 2, 0))
    return img

def to_gray_uint8(img):
    if img.ndim == 3:
        if img.shape[2] == 3:
            bgr = img
        elif img.shape[2] == 4:
            bgr = img[:, :, :3]
        else:
            bgr = img[:, :, :3]
        if bgr.dtype == np.uint8:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            b = (bgr.astype(np.float32) - np.min(bgr)) / (np.ptp(bgr) + 1e-9)
            b = (b * 255.0).astype(np.uint8)
            gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    else:
        a = img
        if a.dtype == np.uint8:
            gray = a.copy()
        else:
            a = (a.astype(np.float32) - np.min(a)) / (np.ptp(a) + 1e-9)
            gray = (a * 255.0).astype(np.uint8)
    return gray

def find_crop_coordinates(original, cropped, max_coarse_dim=1600):

    orig_gray = to_gray_uint8(original)
    crop_gray = to_gray_uint8(cropped)

    oh, ow = orig_gray.shape[:2]
    ch, cw = crop_gray.shape[:2]

    if ch > oh or cw > ow:
        raise ValueError("Cropped image is larger than original — check inputs.")

    scale = 1.0
    max_dim = max(oh, ow)
    if max_dim > max_coarse_dim:
        scale = max_coarse_dim / float(max_dim)

    if scale < 1.0:
        ow_small = max(1, int(ow * scale))
        oh_small = max(1, int(oh * scale))
        cw_small = max(1, int(cw * scale))
        ch_small = max(1, int(ch * scale))

        orig_small = cv2.resize(orig_gray, (ow_small, oh_small), interpolation=cv2.INTER_AREA)
        crop_small = cv2.resize(crop_gray, (cw_small, ch_small), interpolation=cv2.INTER_AREA)
    else:
        orig_small = orig_gray
        crop_small = crop_gray
        ow_small, oh_small = ow, oh
        cw_small, ch_small = cw, ch

    print("Performing coarse template match at scale {:.4f} (image -> {}x{})".format(scale, ow_small, oh_small))
    res = cv2.matchTemplate(orig_small, crop_small, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    print(f"Coarse match score: {max_val:.4f}, coarse location (small image coords): {max_loc}")

    approx_x = int(round(max_loc[0] / scale))
    approx_y = int(round(max_loc[1] / scale))
    print(f"Approx. top-left in original image: (x={approx_x}, y={approx_y})")


    pad_x = max(50, int(cw * 0.08))
    pad_y = max(50, int(ch * 0.08))

    win_x0 = max(0, approx_x - pad_x)
    win_y0 = max(0, approx_y - pad_y)
    win_x1 = min(ow, approx_x + pad_x + cw + pad_x)
    win_y1 = min(oh, approx_y + pad_y + ch + pad_y)

    search_window = orig_gray[win_y0:win_y1, win_x0:win_x1]
    print(f"Refinement window shape: {search_window.shape}, window top-left: ({win_x0},{win_y0})")

    if search_window.shape[0] < ch or search_window.shape[1] < cw:
        print("Refinement window is smaller than crop. Using coarse approx.")
        final_x, final_y = approx_x, approx_y
    else:
        res2 = cv2.matchTemplate(search_window, crop_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val2, _, max_loc2 = cv2.minMaxLoc(res2)
        final_x = win_x0 + max_loc2[0]
        final_y = win_y0 + max_loc2[1]
        print(f"Refined match score: {max_val2:.4f}, refined top-left: (x={final_x}, y={final_y})")

    final_x = int(max(0, min(final_x, ow - cw)))
    final_y = int(max(0, min(final_y, oh - ch)))

    return final_x, final_y

def main():
    print("Loading images (this can take some time for very large TIFFs)...")
    original = read_image(original_path)
    cropped = read_image(cropped_path)
    print(f"Original shape: {original.shape}, dtype: {original.dtype}")
    print(f"Cropped  shape: {cropped.shape}, dtype: {cropped.dtype}")

    x, y = find_crop_coordinates(original, cropped, max_coarse_dim=1600)
    print(f"Final crop coordinates in original: top-left = ({x},{y}), size = ({cropped.shape[1]} x {cropped.shape[0]})")

    h_crop, w_crop = cropped.shape[0], cropped.shape[1]
    extracted = original[y:y+h_crop, x:x+w_crop].copy()

    if extracted.shape[0] != h_crop or extracted.shape[1] != w_crop:
        raise RuntimeError("Extracted crop size mismatch — something went wrong.")

    print(f"Saving extracted crop to: {output_path}")
    imwrite(output_path, extracted)
    print("Saved. Done.")

if __name__ == "__main__":
    main()
