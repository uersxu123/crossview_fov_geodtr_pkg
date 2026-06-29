# -*- coding: utf-8 -*-
"""Phone-image mask refinement driven by coarse satellite projection masks."""

import cv2
import numpy as np

import config as C
from .sat_to_phone import clean_phone_mask


def _odd(value, minimum=3):
    value = int(round(value))
    value = max(minimum, value)
    return value | 1


def _resize_to_max(img, masks, max_dim):
    h, w = img.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale >= 1.0:
        return img.copy(), [m.copy() for m in masks], 1.0
    size = (int(round(w * scale)), int(round(h * scale)))
    small_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    small_masks = [cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) for m in masks]
    return small_img, small_masks, scale


def _remove_small(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros(mask.shape[:2], dtype=np.uint8)
    for idx in range(1, n):
        if stats[idx, cv2.CC_STAT_AREA] >= int(min_area):
            out[labels == idx] = 255
    return out


def _fill_holes(mask):
    h, w = mask.shape[:2]
    flood = (mask > 0).astype(np.uint8) * 255
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or((mask > 0).astype(np.uint8) * 255, holes)


def _phone_color_masks(phone_img):
    hsv = cv2.cvtColor(phone_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(phone_img, cv2.COLOR_BGR2LAB)
    h, s, v = cv2.split(hsv)
    l, a, _ = cv2.split(lab)

    green = ((h >= 32) & (h <= 95) & (s > 28) & (v > 35)).astype(np.uint8) * 255
    yellow_or_orange = ((h >= 8) & (h <= 36) & (s > 80) & (v > 80))
    road = ((s < 95) & (v > 25) & (l > 25) & (a >= 105) & (a <= 155) & ~yellow_or_orange).astype(np.uint8) * 255
    return road, green


def _run_grabcut(img, probable_fg, sure_fg, sure_bg, iters):
    gc_mask = np.full(img.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[probable_fg > 0] = cv2.GC_PR_FGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD
    gc_mask[sure_bg > 0] = cv2.GC_BGD

    fg_count = int(np.count_nonzero((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)))
    bg_count = int(np.count_nonzero(gc_mask == cv2.GC_BGD))
    if fg_count < 40 or bg_count < 40:
        return (probable_fg > 0).astype(np.uint8) * 255

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, gc_mask, None, bgd, fgd, int(iters), cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return (probable_fg > 0).astype(np.uint8) * 255
    return np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def refine_mask_with_grabcut(phone_img, coarse_mask, negative_mask=None, min_area=None, kind=None):
    """Refine one coarse phone-space mask with strict GrabCut seeds."""
    max_dim = int(getattr(C, "PHONE_REFINE_MAX_DIM", 1200))
    negative_mask = np.zeros(coarse_mask.shape[:2], dtype=np.uint8) if negative_mask is None else negative_mask
    img, resized, scale = _resize_to_max(phone_img, [coarse_mask, negative_mask], max_dim)
    coarse, negative = resized

    coarse = (coarse > 0).astype(np.uint8) * 255
    if np.count_nonzero(coarse) < 50:
        return coarse_mask.copy()

    fg_erode_k = _odd(getattr(C, "PHONE_REFINE_SURE_FG_ERODE_K", 17))
    fg_dilate_k = _odd(getattr(C, "PHONE_REFINE_PROBABLE_FG_DILATE_K", 35))
    bg_dilate_k = _odd(getattr(C, "PHONE_REFINE_SURE_BG_DILATE_K", 75))

    sure_fg = cv2.erode(coarse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fg_erode_k, fg_erode_k)))
    probable_fg = cv2.dilate(coarse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fg_dilate_k, fg_dilate_k)))
    far_fg = cv2.dilate(coarse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_dilate_k, bg_dilate_k)))
    sure_bg = cv2.bitwise_not(far_fg)
    sure_bg[negative > 0] = 255
    sure_fg[negative > 0] = 0
    probable_fg[negative > 0] = 0

    road_color, green_color = _phone_color_masks(img)
    green_support = cv2.dilate(green_color, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    if kind == "green" and np.count_nonzero(cv2.bitwise_and(probable_fg, green_support)) > 50:
        probable_fg = cv2.bitwise_and(probable_fg, green_support)
        sure_fg = cv2.bitwise_and(sure_fg, green_support)
        sure_bg = cv2.bitwise_or(sure_bg, cv2.bitwise_not(cv2.dilate(green_support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))))
    elif kind == "road":
        green_strong = cv2.erode(green_color, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        sure_bg = cv2.bitwise_or(sure_bg, green_strong)
        probable_fg[green_strong > 0] = 0
        sure_fg[green_strong > 0] = 0

    refined = _run_grabcut(
        img,
        probable_fg,
        sure_fg,
        sure_bg,
        getattr(C, "PHONE_REFINE_GRABCUT_ITERS", 4),
    )
    refined = cv2.bitwise_and(refined, probable_fg)
    refined[negative > 0] = 0
    if kind == "green" and np.count_nonzero(green_support) > 50:
        refined = cv2.bitwise_and(refined, green_support)
    elif kind == "road":
        refined[green_color > 0] = 0

    close_k = _odd(getattr(C, "PHONE_REFINE_CLOSE_K", 17))
    open_k = _odd(getattr(C, "PHONE_REFINE_OPEN_K", 5))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    refined = _fill_holes(refined)
    refined = _remove_small(refined, min_area or getattr(C, "PHONE_REFINE_MIN_AREA", 900))

    if scale < 1.0:
        refined = cv2.resize(refined, (phone_img.shape[1], phone_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return clean_phone_mask(
        refined,
        min_area=min_area or getattr(C, "PHONE_REFINE_MIN_AREA", 900),
        close_k=getattr(C, "PHONE_REFINE_FINAL_CLOSE_K", 9),
        open_k=getattr(C, "PHONE_REFINE_FINAL_OPEN_K", 3),
        ground_only=True,
    )


def refine_sat_projected_masks(phone_img, coarse_road, coarse_green):
    """Refine road and green masks without SAM, using only image evidence and GrabCut."""
    coarse_road = (coarse_road > 0).astype(np.uint8) * 255
    coarse_green = (coarse_green > 0).astype(np.uint8) * 255

    road = refine_mask_with_grabcut(phone_img, coarse_road, negative_mask=coarse_green, kind="road")
    green = refine_mask_with_grabcut(phone_img, coarse_green, negative_mask=road, kind="green")
    green[road > 0] = 0
    return {"road_mask": road, "green_mask": green}


def draw_phone_mask_overlay(phone_img, road_mask, green_mask, alpha=None):
    """Draw refined masks on the phone image."""
    alpha = float(alpha if alpha is not None else getattr(C, "PHONE_REFINE_VIS_ALPHA", 0.42))
    out = phone_img.copy()
    road_color = np.zeros_like(out)
    road_color[:, :, 0] = 255
    green_color = np.zeros_like(out)
    green_color[:, :, 1] = 230
    green_color[:, :, 2] = 255

    road_blend = cv2.addWeighted(out, 1.0 - alpha, road_color, alpha, 0)
    green_blend = cv2.addWeighted(out, 1.0 - alpha, green_color, alpha, 0)
    out[green_mask > 0] = green_blend[green_mask > 0]
    out[road_mask > 0] = road_blend[road_mask > 0]

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 200, 255), 3)
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 80, 0), 3)
    return out
