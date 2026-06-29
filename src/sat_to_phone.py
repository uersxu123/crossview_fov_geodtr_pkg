# -*- coding: utf-8 -*-
"""Project satellite road/green masks back into the phone image."""

import cv2
import numpy as np

import config as C
from .mask_projection import _phone_uv_to_sat_vectorized


def _odd(value, minimum=3):
    value = int(round(value))
    value = max(minimum, value)
    return value | 1


def _sample_sat_mask_nearest(mask, sat_pts):
    """Nearest-neighbor sample a satellite binary mask at floating-point xy coordinates."""
    mh, mw = mask.shape[:2]
    xs = np.rint(sat_pts[:, 0]).astype(np.int32)
    ys = np.rint(sat_pts[:, 1]).astype(np.int32)
    valid = (xs >= 0) & (xs < mw) & (ys >= 0) & (ys < mh)

    out = np.zeros((sat_pts.shape[0],), dtype=np.uint8)
    out[valid] = (mask[ys[valid], xs[valid]] > 0).astype(np.uint8) * 255
    return out


def _ground_roi(phone_shape):
    """Keep only the likely ground/lower-image region to reduce sky/building false positives."""
    h, w = phone_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    top = int(h * float(getattr(C, "PHONE_GROUND_TOP_RATIO", 0.43)))
    bottom = int(h * float(getattr(C, "PHONE_GROUND_BOTTOM_RATIO", 0.98)))
    left = int(w * float(getattr(C, "PHONE_SIDE_MARGIN_RATIO", 0.04)))
    right = int(w * (1.0 - float(getattr(C, "PHONE_SIDE_MARGIN_RATIO", 0.04))))
    mask[top:bottom, left:right] = 255
    return mask


def _remove_small_components(mask, min_area):
    min_area = int(min_area)
    if min_area <= 0:
        return mask

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def clean_phone_mask(mask, min_area=None, close_k=None, open_k=None, ground_only=None):
    """Clean a satellite-guided phone-space mask with ROI and morphology filters."""
    min_area = int(min_area if min_area is not None else getattr(C, "SAT_TO_PHONE_MIN_AREA", 800))
    close_k = _odd(close_k if close_k is not None else getattr(C, "SAT_TO_PHONE_CLOSE_K", 15))
    open_k = _odd(open_k if open_k is not None else getattr(C, "SAT_TO_PHONE_OPEN_K", 5))
    ground_only = bool(ground_only if ground_only is not None else getattr(C, "SAT_TO_PHONE_GROUND_ONLY", True))

    out = (mask > 0).astype(np.uint8) * 255
    if ground_only:
        out = cv2.bitwise_and(out, _ground_roi(out.shape))
    if close_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    if open_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    return _remove_small_components(out, min_area)


def project_sat_mask_to_phone(sat_mask, pose, phone_shape, sample_step=None, clean=True):
    """Map one satellite-space mask into phone-image coordinates using the current pose."""
    ph, pw = phone_shape[:2]
    sample_step = int(sample_step or getattr(C, "SAT_TO_PHONE_SAMPLE_STEP", 2))
    sample_step = max(1, sample_step)

    ys, xs = np.mgrid[0:ph:sample_step, 0:pw:sample_step]
    uv = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    sat_pts = _phone_uv_to_sat_vectorized(uv, pose, phone_shape)
    sampled = _sample_sat_mask_nearest(sat_mask, sat_pts).reshape(ys.shape)

    if sample_step == 1:
        phone_mask = sampled
    else:
        phone_mask = cv2.resize(sampled, (pw, ph), interpolation=cv2.INTER_NEAREST)

    if clean:
        phone_mask = clean_phone_mask(phone_mask)
    return phone_mask


def project_sat_feature_masks_to_phone(sat_feat, pose, phone_shape, sample_step=None):
    """Project satellite road and green masks back to the phone image."""
    road = project_sat_mask_to_phone(sat_feat["road_mask"], pose, phone_shape, sample_step=sample_step, clean=True)
    green = project_sat_mask_to_phone(sat_feat["green_mask"], pose, phone_shape, sample_step=sample_step, clean=True)
    green[road > 0] = 0
    return {"road_mask": road, "green_mask": green}


def draw_sat_guided_phone_overlay(phone_img, road_mask, green_mask):
    """Visualize satellite-guided road/green masks on the phone image."""
    out = phone_img.copy()
    alpha = float(getattr(C, "SAT_TO_PHONE_VIS_ALPHA", 0.42))

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
