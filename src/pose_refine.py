# -*- coding: utf-8 -*-
import csv
import math
import random
from dataclasses import dataclass
import numpy as np
import cv2
import config as C
from .geometry import Pose, angle_diff_deg, project_uv_to_sat, build_fov_polygon, vector_from_bearing, draw_alpha_poly, clamp

@dataclass
class Result:
    pose: Pose
    score: float
    parts: dict
    retrieval_score: float


def yaw_search_center():
    return (C.YAW_INIT_DEG + getattr(C, "YAW_BIAS_DEG", 0.0)) % 360


def build_uv_samples(phone_img):
    h, w = phone_img.shape[:2]
    rng = np.random.default_rng(C.RANDOM_SEED)
    umin = w * C.PHONE_SIDE_MARGIN_RATIO
    umax = w * (1.0 - C.PHONE_SIDE_MARGIN_RATIO)
    vmin = h * C.PHONE_GROUND_TOP_RATIO
    vmax = h * C.PHONE_GROUND_BOTTOM_RATIO
    ground = np.stack([rng.uniform(umin, umax, 220), rng.uniform(vmin, vmax, 220)], axis=1).astype(np.float32)
    center = np.stack([np.full(80, w * 0.5), np.linspace(vmin, vmax, 80)], axis=1).astype(np.float32)
    vs = np.linspace(vmin, vmax, 60)
    boundary = np.vstack([np.stack([np.full_like(vs, umin), vs], axis=1), np.stack([np.full_like(vs, umax), vs], axis=1)]).astype(np.float32)
    return ground, center, boundary


def sample_score_map(score_map, pts):
    h, w = score_map.shape[:2]
    xs = np.rint(pts[:,0]).astype(np.int32)
    ys = np.rint(pts[:,1]).astype(np.int32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    vals = np.zeros(len(pts), dtype=np.float32)
    vals[valid] = score_map[ys[valid], xs[valid]]
    return vals, valid


def shape_penalty(poly, sat_shape):
    sh, sw = sat_shape[:2]
    area = abs(cv2.contourArea(poly.astype(np.float32)))
    ratio = area / max(sw * sh, 1)
    penalty = 0.0
    if ratio > 0.55:
        penalty += (ratio - 0.55) / 0.45
    if ratio < 0.002:
        penalty += (0.002 - ratio) / 0.002
    clipped = np.sum((poly[:,0] <= 1) | (poly[:,0] >= sw-2) | (poly[:,1] <= 1) | (poly[:,1] >= sh-2))
    penalty += 0.15 * clipped
    return float(np.clip(penalty, 0, 2))


def eval_pose(pose, sat_feat, phone_shape, sat_shape, uv_samples, retrieval_score):
    ground_uv, center_uv, boundary_uv = uv_samples
    ground = project_uv_to_sat(ground_uv, pose, phone_shape, sat_shape)
    center = project_uv_to_sat(center_uv, pose, phone_shape, sat_shape)
    boundary = project_uv_to_sat(boundary_uv, pose, phone_shape, sat_shape)
    poly = build_fov_polygon(pose, phone_shape, sat_shape)

    road_vals, valid_g = sample_score_map(sat_feat["road_score"], ground)
    center_vals, valid_c = sample_score_map(sat_feat["road_score"], center)
    edge_vals, _ = sample_score_map(sat_feat["edge_score"], boundary)
    bad_vals, _ = sample_score_map(sat_feat["bad_score"], ground)

    road = float(np.mean(road_vals))
    center_road = float(np.mean(center_vals))
    boundary_edge = float(np.mean(edge_vals))
    bad = float(np.mean(bad_vals))
    gps_dist = math.hypot(pose.x - C.GPS_XY_INIT[0], pose.y - C.GPS_XY_INIT[1])
    gps_prior = math.exp(-0.5 * (gps_dist / max(C.SEARCH_XY_RADIUS_PX * 0.6, 1)) ** 2)
    yaw_prior = math.exp(-0.5 * (angle_diff_deg(pose.yaw, yaw_search_center()) / max(C.SEARCH_YAW_RADIUS_DEG * 0.7, 1)) ** 2)
    shp = shape_penalty(poly, sat_shape)
    out_penalty = 1.0 - float(np.mean(valid_g.astype(np.float32)))

    score = (C.W_DEEP_RETRIEVAL * retrieval_score + C.W_ROAD_OVERLAP * road + C.W_CENTERLINE_ROAD * center_road +
             C.W_BOUNDARY_EDGE * boundary_edge + C.W_GPS_PRIOR * gps_prior + C.W_YAW_PRIOR * yaw_prior -
             C.W_BAD_REGION * bad - C.W_SHAPE * shp - 0.35 * out_penalty)
    parts = {"road": road, "center_road": center_road, "boundary_edge": boundary_edge, "bad": bad,
             "gps_dist": gps_dist, "gps_prior": gps_prior, "yaw_prior": yaw_prior, "shape": shp, "out": out_penalty}
    return float(score), parts


def random_pose_around(cx, cy, rng):
    gx, gy = C.GPS_XY_INIT
    # 以检索 patch center 和 GPS 共同约束，避免跑飞；必要时直接锁死到 GPS。
    if getattr(C, "LOCK_CAMERA_TO_GPS", False):
        x, y = gx, gy
    else:
        x = rng.gauss(cx, C.SEARCH_XY_RADIUS_PX * 0.35)
        y = rng.gauss(cy, C.SEARCH_XY_RADIUS_PX * 0.35)
        x = clamp(x, gx - C.SEARCH_XY_RADIUS_PX, gx + C.SEARCH_XY_RADIUS_PX)
        y = clamp(y, gy - C.SEARCH_XY_RADIUS_PX, gy + C.SEARCH_XY_RADIUS_PX)
    yaw = (yaw_search_center() + rng.uniform(-C.SEARCH_YAW_RADIUS_DEG, C.SEARCH_YAW_RADIUS_DEG)) % 360
    pitch = rng.uniform(C.PITCH_MIN_DEG, C.PITCH_MAX_DEG)
    hfov = rng.uniform(C.HFOV_MIN_DEG, C.HFOV_MAX_DEG)
    mpp = rng.uniform(C.MPP_MIN, C.MPP_MAX)
    return Pose(x, y, yaw, pitch, hfov, mpp)


def refine_pose(phone_img, sat_img, sat_feat, top_patches):
    rng = random.Random(C.RANDOM_SEED)
    uv_samples = build_uv_samples(phone_img)
    results = []
    for rank, p in enumerate(top_patches, start=1):
        base_n = C.CANDIDATES_PER_PATCH if rank <= 10 else max(60, C.CANDIDATES_PER_PATCH // 3)
        for _ in range(base_n):
            pose = random_pose_around(p["x"], p["y"], rng)
            score, parts = eval_pose(pose, sat_feat, phone_img.shape, sat_img.shape, uv_samples, p["score"])
            results.append(Result(pose, score, parts, p["score"]))

    # local refinement around current best
    results.sort(key=lambda r: r.score, reverse=True)
    parents = results[:30]
    for _round in range(C.LOCAL_REFINE_ROUNDS):
        new = []
        for r in parents:
            for _ in range(35):
                pose = Pose(
                    x=r.pose.x + rng.gauss(0, 6.0),
                    y=r.pose.y + rng.gauss(0, 6.0),
                    yaw=(r.pose.yaw + rng.gauss(0, 3.0)) % 360,
                    pitch=clamp(r.pose.pitch + rng.gauss(0, 1.2), C.PITCH_MIN_DEG, C.PITCH_MAX_DEG),
                    hfov=clamp(r.pose.hfov + rng.gauss(0, 2.0), C.HFOV_MIN_DEG, C.HFOV_MAX_DEG),
                    mpp=clamp(r.pose.mpp + rng.gauss(0, 0.025), C.MPP_MIN, C.MPP_MAX)
                )
                # keep inside GPS radius
                gx, gy = C.GPS_XY_INIT
                if getattr(C, "LOCK_CAMERA_TO_GPS", False):
                    pose.x, pose.y = gx, gy
                else:
                    pose.x = clamp(pose.x, gx - C.SEARCH_XY_RADIUS_PX, gx + C.SEARCH_XY_RADIUS_PX)
                    pose.y = clamp(pose.y, gy - C.SEARCH_XY_RADIUS_PX, gy + C.SEARCH_XY_RADIUS_PX)
                score, parts = eval_pose(pose, sat_feat, phone_img.shape, sat_img.shape, uv_samples, r.retrieval_score)
                new.append(Result(pose, score, parts, r.retrieval_score))
        results.extend(new)
        results.sort(key=lambda r: r.score, reverse=True)
        parents = results[:30]
    return results[:C.SAVE_TOPK_POSES]


def draw_best_overlay(sat_img, phone_img, results):
    best = results[0]
    out = sat_img.copy()
    # draw top candidate centerlines in gray
    for r in results[:15]:
        uv = np.array([[phone_img.shape[1]*0.5, phone_img.shape[0]*C.PHONE_GROUND_TOP_RATIO],
                       [phone_img.shape[1]*0.5, phone_img.shape[0]*C.PHONE_GROUND_BOTTOM_RATIO]], dtype=np.float32)
        pts = project_uv_to_sat(uv, r.pose, phone_img.shape, sat_img.shape)
        pts[:,0] = np.clip(pts[:,0], 0, sat_img.shape[1]-1)
        pts[:,1] = np.clip(pts[:,1], 0, sat_img.shape[0]-1)
        cv2.line(out, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), (80,80,80), 1)

    poly = build_fov_polygon(best.pose, phone_img.shape, sat_img.shape)
    out = draw_alpha_poly(out, poly, (255,100,0), 0.28)
    cv2.polylines(out, [poly.astype(np.int32)], True, (0,0,255), 3)
    cam = np.array([best.pose.x, best.pose.y])
    cv2.line(out, tuple(cam.astype(int)), tuple(poly[0].astype(int)), (0,0,255), 3)
    cv2.line(out, tuple(cam.astype(int)), tuple(poly[1].astype(int)), (0,0,255), 3)
    center = cam + vector_from_bearing(best.pose.yaw) * (C.MAX_RANGE_M / best.pose.mpp)
    center[0] = np.clip(center[0], 0, sat_img.shape[1]-1); center[1] = np.clip(center[1], 0, sat_img.shape[0]-1)
    cv2.line(out, tuple(cam.astype(int)), tuple(center.astype(int)), (0,255,255), 3)
    gps = np.array(C.GPS_XY_INIT)
    cv2.circle(out, tuple(gps.astype(int)), 11, (255,0,255), 2)
    cv2.circle(out, tuple(cam.astype(int)), 8, (255,180,0), -1)
    cv2.circle(out, tuple(cam.astype(int)), 9, (255,255,255), 2)
    txt1 = f"score={best.score:.3f}, yaw={best.pose.yaw:.1f}, yaw_center={yaw_search_center():.1f}, pitch={best.pose.pitch:.1f}, hfov={best.pose.hfov:.1f}, mpp={best.pose.mpp:.3f}"
    txt2 = f"camera=({best.pose.x:.1f},{best.pose.y:.1f}), gps_dist={best.parts['gps_dist']:.1f}, retrieval={best.retrieval_score:.3f}, road={best.parts['road']:.2f}"
    cv2.rectangle(out, (8,8), (1050,72), (0,0,0), -1)
    cv2.putText(out, txt1, (16,35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)
    cv2.putText(out, txt2, (16,60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
    return out


def save_results_csv(path, results):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        fields = ["rank","score","retrieval","x","y","yaw","pitch","hfov","mpp","road","center_road","boundary_edge","bad","gps_dist","gps_prior","yaw_prior","shape","out"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for i, r in enumerate(results, start=1):
            row = {"rank": i, "score": r.score, "retrieval": r.retrieval_score,
                   "x": r.pose.x, "y": r.pose.y, "yaw": r.pose.yaw, "pitch": r.pose.pitch, "hfov": r.pose.hfov, "mpp": r.pose.mpp}
            row.update(r.parts)
            writer.writerow(row)
