# -*- coding: utf-8 -*-
import math

import cv2
import numpy as np

import config as C
from .geometry import right_vector_from_bearing, vector_from_bearing


def _phone_uv_to_sat_vectorized(uv, pose, phone_shape):
    ph, pw = phone_shape[:2]
    aspect = ph / max(pw, 1)
    vfov = math.degrees(2.0 * math.atan(math.tan(math.radians(pose.hfov) / 2.0) * aspect))
    fx = (pw / 2.0) / math.tan(math.radians(pose.hfov) / 2.0)
    fy = (ph / 2.0) / math.tan(math.radians(vfov) / 2.0)
    cx = pw / 2.0
    cy = ph / 2.0

    u = uv[:, 0].astype(np.float64)
    v = uv[:, 1].astype(np.float64)
    x = (u - cx) / fx
    y = np.ones_like(x)
    z = -(v - cy) / fy

    pitch = math.radians(pose.pitch)
    y2 = y * math.cos(pitch) + z * math.sin(pitch)
    z2 = -y * math.sin(pitch) + z * math.cos(pitch)
    x2 = x

    gx = np.zeros_like(x2)
    gy = np.zeros_like(y2)
    horizon = z2 >= -1e-6

    if np.any(~horizon):
        t = -float(C.CAMERA_HEIGHT_M) / z2[~horizon]
        gx[~horizon] = x2[~horizon] * t
        gy[~horizon] = y2[~horizon] * t

    if np.any(horizon):
        n = np.hypot(x2[horizon], y2[horizon])
        n = np.maximum(n, 1e-8)
        s = float(C.MAX_RANGE_M) / n
        gx[horizon] = x2[horizon] * s
        gy[horizon] = y2[horizon] * s

    d = np.hypot(gx, gy)
    near = d < float(C.NEAR_RANGE_M)
    if np.any(near):
        s = float(C.NEAR_RANGE_M) / np.maximum(d[near], 1e-8)
        gx[near] *= s
        gy[near] *= s

    far = d > float(C.MAX_RANGE_M)
    if np.any(far):
        s = float(C.MAX_RANGE_M) / np.maximum(d[far], 1e-8)
        gx[far] *= s
        gy[far] *= s

    fwd = vector_from_bearing(pose.yaw)
    right = right_vector_from_bearing(pose.yaw)
    origin = np.array([pose.x, pose.y], dtype=np.float64)
    sat = origin + (right.reshape(1, 2) * gx[:, None] + fwd.reshape(1, 2) * gy[:, None]) / pose.mpp
    return sat


def project_phone_mask_to_sat(mask, pose, phone_shape, sat_shape, sample_step=None, dilate_px=None):
    sample_step = int(sample_step or getattr(C, "PHONE_TO_SAT_SAMPLE_STEP", 2))
    dilate_px = int(dilate_px or getattr(C, "PHONE_TO_SAT_DILATE_PX", 3))
    sh, sw = sat_shape[:2]
    h, w = mask.shape[:2]
    ys, xs = np.mgrid[0:h:sample_step, 0:w:sample_step]
    keep = mask[ys, xs] > 0
    out = np.zeros((sh, sw), dtype=np.uint8)
    if not np.any(keep):
        return out

    uv = np.stack([xs[keep], ys[keep]], axis=1).astype(np.float32)
    sat_pts = _phone_uv_to_sat_vectorized(uv, pose, phone_shape)
    sx = np.rint(sat_pts[:, 0]).astype(np.int32)
    sy = np.rint(sat_pts[:, 1]).astype(np.int32)
    valid = (sx >= 0) & (sx < sw) & (sy >= 0) & (sy < sh)
    out[sy[valid], sx[valid]] = 255

    if dilate_px > 0:
        ksize = max(3, dilate_px * 2 + 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        out = cv2.dilate(out, k, iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
        bridge_w = max(3, int(getattr(C, "PHONE_TO_SAT_CLOSE_W", 9)) | 1)
        bridge_h = max(3, int(getattr(C, "PHONE_TO_SAT_CLOSE_H", 21)) | 1)
        bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_w, bridge_h))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, bridge)
    return out


def draw_projected_masks_on_sat(sat_img, road_mask, green_mask):
    out = sat_img.copy()
    road_color = np.zeros_like(out)
    road_color[:, :, 0] = 255
    green_color = np.zeros_like(out)
    green_color[:, :, 1] = 230
    green_color[:, :, 2] = 255

    road_blend = cv2.addWeighted(out, 0.55, road_color, 0.45, 0)
    green_blend = cv2.addWeighted(out, 0.58, green_color, 0.42, 0)
    out[green_mask > 0] = green_blend[green_mask > 0]
    out[road_mask > 0] = road_blend[road_mask > 0]

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 200, 255), 2)
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 80, 0), 2)
    return out


def projected_area_m2(mask, pose):
    return float(np.count_nonzero(mask)) * float(pose.mpp) * float(pose.mpp)
