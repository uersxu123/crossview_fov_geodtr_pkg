# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
import numpy as np
import cv2
import config as C

@dataclass
class Pose:
    x: float
    y: float
    yaw: float
    pitch: float
    hfov: float
    mpp: float


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def angle_diff_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def vector_from_bearing(deg):
    t = math.radians(deg)
    return np.array([math.sin(t), -math.cos(t)], dtype=np.float64)


def right_vector_from_bearing(deg):
    return vector_from_bearing(deg + 90.0)


def pixel_ray_to_ground_m(u, v, img_w, img_h, hfov_deg, pitch_down_deg):
    aspect = img_h / max(img_w, 1)
    vfov_deg = math.degrees(2.0 * math.atan(math.tan(math.radians(hfov_deg) / 2.0) * aspect))

    fx = (img_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = (img_h / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    cx = img_w / 2.0
    cy = img_h / 2.0

    # camera coordinates: y forward, x right, z up. Image down -> negative z.
    x = (u - cx) / fx
    y = 1.0
    z = -(v - cy) / fy

    pitch = math.radians(pitch_down_deg)
    y2 = y * math.cos(pitch) + z * math.sin(pitch)
    z2 = -y * math.sin(pitch) + z * math.cos(pitch)
    x2 = x

    if z2 >= -1e-6:
        n = math.hypot(x2, y2)
        if n < 1e-8:
            return np.array([0.0, C.MAX_RANGE_M], dtype=np.float64)
        s = C.MAX_RANGE_M / n
        return np.array([x2 * s, y2 * s], dtype=np.float64)

    t = -C.CAMERA_HEIGHT_M / z2
    gx = x2 * t
    gy = y2 * t
    d = math.hypot(gx, gy)

    if d < C.NEAR_RANGE_M:
        s = C.NEAR_RANGE_M / max(d, 1e-8)
        gx *= s
        gy *= s
    if d > C.MAX_RANGE_M:
        s = C.MAX_RANGE_M / max(d, 1e-8)
        gx *= s
        gy *= s
    return np.array([gx, gy], dtype=np.float64)


def project_uv_to_sat(uv_points, pose: Pose, phone_shape, sat_shape):
    ph, pw = phone_shape[:2]
    fwd = vector_from_bearing(pose.yaw)
    right = right_vector_from_bearing(pose.yaw)
    origin = np.array([pose.x, pose.y], dtype=np.float64)
    out = []
    for u, v in uv_points:
        xy_m = pixel_ray_to_ground_m(float(u), float(v), pw, ph, pose.hfov, pose.pitch)
        offset = (right * xy_m[0] + fwd * xy_m[1]) / pose.mpp
        out.append(origin + offset)
    return np.asarray(out, dtype=np.float64)


def build_fov_polygon(pose: Pose, phone_shape, sat_shape, clip=True):
    ph, pw = phone_shape[:2]
    sh, sw = sat_shape[:2]
    u_left = pw * C.PHONE_SIDE_MARGIN_RATIO
    u_right = pw * (1.0 - C.PHONE_SIDE_MARGIN_RATIO)
    v_top = ph * C.PHONE_GROUND_TOP_RATIO
    v_bottom = ph * C.PHONE_GROUND_BOTTOM_RATIO
    uv = np.array([[u_left, v_top], [u_right, v_top], [u_right, v_bottom], [u_left, v_bottom]], dtype=np.float32)
    pts = project_uv_to_sat(uv, pose, phone_shape, sat_shape)
    if clip:
        pts[:, 0] = np.clip(pts[:, 0], 0, sw - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, sh - 1)
    return pts


def draw_alpha_poly(img, pts, color, alpha):
    out = img.copy()
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts.astype(np.int32)], color)
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
    return out
