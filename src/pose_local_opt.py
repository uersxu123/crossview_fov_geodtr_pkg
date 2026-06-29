# -*- coding: utf-8 -*-
"""Local pose optimization by matching projected satellite masks to phone masks."""

import math
import random

import cv2
import numpy as np

import config as C
from .boundary_metrics import alignment_score
from .geometry import Pose, angle_diff_deg, clamp
from .pose_refine import Result
from .sat_to_phone import project_sat_feature_masks_to_phone


def _resize_mask(mask, scale):
    if scale >= 1.0:
        return mask
    h, w = mask.shape[:2]
    return cv2.resize(mask, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_NEAREST)


def _prepare_targets(target_masks, max_dim):
    road = target_masks["road_mask"]
    green = target_masks["green_mask"]
    h, w = road.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    return {
        "road_mask": _resize_mask(road, scale),
        "green_mask": _resize_mask(green, scale),
    }, scale


def _scaled_phone_shape(phone_shape, scale):
    if scale >= 1.0:
        return phone_shape
    h, w = phone_shape[:2]
    return (int(round(h * scale)), int(round(w * scale))) + tuple(phone_shape[2:])


def _mask_size_terms(pred, target):
    pred_bin = pred > 0
    target_bin = target > 0
    pred_count = int(np.count_nonzero(pred_bin))
    target_count = int(np.count_nonzero(target_bin))
    if target_count == 0:
        return 1.0 if pred_count == 0 else 0.0, 1.0 if pred_count == 0 else 0.0
    if pred_count == 0:
        return 0.0, 0.0

    inter = int(np.count_nonzero(pred_bin & target_bin))
    recall = inter / float(target_count)
    area_similarity = min(pred_count, target_count) / float(max(pred_count, target_count))
    return float(recall), float(area_similarity)


def _combine_mask_score(base_score, metrics, pred, target):
    recall, area_similarity = _mask_size_terms(pred, target)
    score = (
        float(base_score)
        + float(getattr(C, "POSE_LOCAL_RECALL_WEIGHT", 0.35)) * recall
        + float(getattr(C, "POSE_LOCAL_AREA_WEIGHT", 0.25)) * area_similarity
    )
    metrics["recall"] = recall
    metrics["area_similarity"] = area_similarity
    return float(score), metrics


def score_pose_by_mask_alignment(sat_feat, pose, phone_shape, target_masks):
    """Project satellite masks with one pose and score against refined phone masks."""
    max_dim = int(getattr(C, "POSE_LOCAL_MAX_DIM", 900))
    targets, scale = _prepare_targets(target_masks, max_dim)
    eval_shape = _scaled_phone_shape(phone_shape, scale)
    sample_step = int(getattr(C, "POSE_LOCAL_SAMPLE_STEP", 5))

    projected = project_sat_feature_masks_to_phone(sat_feat, pose, eval_shape, sample_step=sample_step)
    road_score, road_metrics = alignment_score(
        projected["road_mask"],
        targets["road_mask"],
        boundary_tolerance=getattr(C, "POSE_LOCAL_BOUNDARY_TOLERANCE", 6),
    )
    road_score, road_metrics = _combine_mask_score(
        road_score,
        road_metrics,
        projected["road_mask"],
        targets["road_mask"],
    )
    green_score, green_metrics = alignment_score(
        projected["green_mask"],
        targets["green_mask"],
        boundary_tolerance=getattr(C, "POSE_LOCAL_BOUNDARY_TOLERANCE", 6),
    )
    green_score, green_metrics = _combine_mask_score(
        green_score,
        green_metrics,
        projected["green_mask"],
        targets["green_mask"],
    )

    road_pixels = max(1, int(np.count_nonzero(targets["road_mask"])))
    green_pixels = max(1, int(np.count_nonzero(targets["green_mask"])))
    road_w = road_pixels / float(road_pixels + green_pixels)
    green_w = 1.0 - road_w
    score = road_w * road_score + green_w * green_score

    return float(score), {
        "local_road_iou": road_metrics["iou"],
        "local_green_iou": green_metrics["iou"],
        "local_road_boundary_f1": road_metrics["boundary_f1"],
        "local_green_boundary_f1": green_metrics["boundary_f1"],
        "local_road_chamfer": road_metrics["chamfer"],
        "local_green_chamfer": green_metrics["chamfer"],
        "local_road_recall": road_metrics["recall"],
        "local_green_recall": green_metrics["recall"],
        "local_road_area_similarity": road_metrics["area_similarity"],
        "local_green_area_similarity": green_metrics["area_similarity"],
        "local_mask_score": float(score),
    }


def _candidate_pose(base, dx, dy, dyaw, dpitch, dhfov, dmpp):
    gx, gy = C.GPS_XY_INIT
    if getattr(C, "LOCK_CAMERA_TO_GPS", False):
        x, y = gx, gy
    else:
        xy_radius = float(getattr(C, "POSE_LOCAL_XY_RADIUS_PX", 15.0))
        x = clamp(base.x + dx, gx - xy_radius, gx + xy_radius)
        y = clamp(base.y + dy, gy - xy_radius, gy + xy_radius)
    return Pose(
        x=x,
        y=y,
        yaw=(base.yaw + dyaw) % 360.0,
        pitch=clamp(base.pitch + dpitch, C.PITCH_MIN_DEG, C.PITCH_MAX_DEG),
        hfov=clamp(base.hfov + dhfov, C.HFOV_MIN_DEG, C.HFOV_MAX_DEG),
        mpp=clamp(base.mpp + dmpp, C.MPP_MIN, C.MPP_MAX),
    )


def _pose_key(pose):
    return (
        round(pose.x, 3),
        round(pose.y, 3),
        round(pose.yaw, 3),
        round(pose.pitch, 3),
        round(pose.hfov, 3),
        round(pose.mpp, 5),
    )


def _local_delta_penalty(pose, base):
    yaw_norm = angle_diff_deg(pose.yaw, base.yaw) / max(float(getattr(C, "POSE_LOCAL_YAW_RADIUS_DEG", 6.0)), 1e-6)
    pitch_norm = abs(pose.pitch - base.pitch) / max(float(getattr(C, "POSE_LOCAL_PITCH_RADIUS_DEG", 4.0)), 1e-6)
    hfov_norm = abs(pose.hfov - base.hfov) / max(float(getattr(C, "POSE_LOCAL_HFOV_RADIUS_DEG", 4.0)), 1e-6)
    mpp_norm = abs(pose.mpp - base.mpp) / max(float(getattr(C, "POSE_LOCAL_MPP_RADIUS", 0.025)), 1e-6)
    xy_norm = math.hypot(pose.x - base.x, pose.y - base.y) / max(float(getattr(C, "POSE_LOCAL_XY_RADIUS_PX", 15.0)), 1e-6)
    penalty = (
        0.25 * min(yaw_norm, 2.0)
        + 0.35 * min(pitch_norm, 2.0)
        + 0.20 * min(hfov_norm, 2.0)
        + 0.15 * min(mpp_norm, 2.0)
        + 0.05 * min(xy_norm, 2.0)
    )
    return float(penalty)


def local_refine_pose_with_phone_mask(best_result, sat_feat, phone_shape, target_masks):
    """Search a small pose neighborhood and return an alignment-refined result."""
    rng = random.Random(getattr(C, "POSE_LOCAL_RANDOM_SEED", getattr(C, "RANDOM_SEED", 42)))
    base = best_result.pose
    xy_radius = float(getattr(C, "POSE_LOCAL_XY_RADIUS_PX", 15.0))
    yaw_radius = float(getattr(C, "POSE_LOCAL_YAW_RADIUS_DEG", 6.0))
    pitch_radius = float(getattr(C, "POSE_LOCAL_PITCH_RADIUS_DEG", 2.5))
    hfov_radius = float(getattr(C, "POSE_LOCAL_HFOV_RADIUS_DEG", 3.0))
    mpp_radius = float(getattr(C, "POSE_LOCAL_MPP_RADIUS", 0.025))
    rounds = [
        (xy_radius, yaw_radius, pitch_radius, hfov_radius, mpp_radius),
        (xy_radius * 0.45, yaw_radius * 0.42, pitch_radius * 0.45, hfov_radius * 0.50, mpp_radius * 0.48),
        (xy_radius * 0.20, yaw_radius * 0.17, pitch_radius * 0.22, hfov_radius * 0.20, mpp_radius * 0.24),
    ]
    per_round = int(getattr(C, "POSE_LOCAL_CANDIDATES_PER_ROUND", 70))

    scored = []
    seen = set()

    def add_pose(pose):
        key = _pose_key(pose)
        if key in seen:
            return
        seen.add(key)
        local_score, metrics = score_pose_by_mask_alignment(sat_feat, pose, phone_shape, target_masks)
        delta_penalty = _local_delta_penalty(pose, base)
        adjusted_score = local_score - float(getattr(C, "POSE_LOCAL_DELTA_PENALTY_WEIGHT", 0.18)) * delta_penalty
        parts = dict(best_result.parts)
        parts.update(metrics)
        parts["gps_dist"] = math.hypot(pose.x - C.GPS_XY_INIT[0], pose.y - C.GPS_XY_INIT[1])
        parts["local_delta_penalty"] = delta_penalty
        parts["local_adjusted_mask_score"] = adjusted_score
        result = Result(
            pose=pose,
            score=float(best_result.score + float(getattr(C, "POSE_LOCAL_SCORE_WEIGHT", 0.35)) * adjusted_score),
            parts=parts,
            retrieval_score=best_result.retrieval_score,
        )
        scored.append(result)

    add_pose(base)
    parents = [base]
    for xy_r, yaw_r, pitch_r, hfov_r, mpp_r in rounds:
        next_parents = []
        for parent in parents:
            for dyaw in (-yaw_r, -yaw_r * 0.5, 0.0, yaw_r * 0.5, yaw_r):
                add_pose(_candidate_pose(parent, 0.0, 0.0, dyaw, 0.0, 0.0, 0.0))
            for dpitch in (-pitch_r, -pitch_r * 0.5, pitch_r * 0.5, pitch_r):
                add_pose(_candidate_pose(parent, 0.0, 0.0, 0.0, dpitch, 0.0, 0.0))
            for dhfov in (-hfov_r, -hfov_r * 0.5, hfov_r * 0.5, hfov_r):
                add_pose(_candidate_pose(parent, 0.0, 0.0, 0.0, 0.0, dhfov, 0.0))

            if not getattr(C, "LOCK_CAMERA_TO_GPS", False):
                for dx, dy in ((-xy_r, 0.0), (xy_r, 0.0), (0.0, -xy_r), (0.0, xy_r)):
                    add_pose(_candidate_pose(parent, dx, dy, 0.0, 0.0, 0.0, 0.0))

            for _ in range(per_round):
                dx = rng.uniform(-xy_r, xy_r) if not getattr(C, "LOCK_CAMERA_TO_GPS", False) else 0.0
                dy = rng.uniform(-xy_r, xy_r) if not getattr(C, "LOCK_CAMERA_TO_GPS", False) else 0.0
                add_pose(
                    _candidate_pose(
                        parent,
                        dx,
                        dy,
                        rng.uniform(-yaw_r, yaw_r),
                        rng.uniform(-pitch_r, pitch_r),
                        rng.uniform(-hfov_r, hfov_r),
                        rng.uniform(-mpp_r, mpp_r),
                    )
                )

        scored.sort(key=lambda r: r.parts.get("local_adjusted_mask_score", -1.0), reverse=True)
        next_parents = [r.pose for r in scored[: int(getattr(C, "POSE_LOCAL_KEEP_PARENTS", 4))]]
        parents = next_parents or parents

    scored.sort(key=lambda r: r.parts.get("local_adjusted_mask_score", -1.0), reverse=True)
    return scored[0], scored[: int(getattr(C, "POSE_LOCAL_SAVE_TOPK", 20))]
