# -*- coding: utf-8 -*-
import numpy as np
import cv2
import config as C


def crop_patch(img, cx, cy, size):
    h, w = img.shape[:2]
    half = size // 2
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    x2, y2 = x1 + size, y1 + size
    patch = np.zeros((size, size, 3), dtype=img.dtype)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    dx1, dy1 = sx1 - x1, sy1 - y1
    patch[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = img[sy1:sy2, sx1:sx2]
    return patch


def generate_satellite_patches(sat_img):
    h, w = sat_img.shape[:2]
    x0, y0 = C.GPS_XY_INIT
    centers = []
    r = C.PATCH_SEARCH_RADIUS
    stride = C.PATCH_STRIDE
    for y in np.arange(y0 - r, y0 + r + 1, stride):
        for x in np.arange(x0 - r, x0 + r + 1, stride):
            if 0 <= x < w and 0 <= y < h:
                if (x - x0) ** 2 + (y - y0) ** 2 <= r ** 2:
                    centers.append((float(x), float(y)))
    patches = [crop_patch(sat_img, x, y, C.PATCH_SIZE) for x, y in centers]
    return centers, patches


def phone_to_query(phone_img):
    # GeoDTR 通常更适配宽幅 ground image。这里保留原图比例后补成方图，避免几何完全变形。
    size = C.PATCH_SIZE
    h, w = phone_img.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(phone_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=phone_img.dtype)
    rh, rw = resized.shape[:2]
    y1 = (size - rh) // 2
    x1 = (size - rw) // 2
    canvas[y1:y1 + rh, x1:x1 + rw] = resized
    return canvas
