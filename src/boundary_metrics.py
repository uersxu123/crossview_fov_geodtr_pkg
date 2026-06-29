# -*- coding: utf-8 -*-
"""Mask alignment and boundary metrics used by local pose refinement."""

import cv2
import numpy as np


def _binary(mask):
    return (mask > 0).astype(np.uint8)


def mask_iou(a, b):
    """Return intersection-over-union for two binary masks."""
    aa = _binary(a)
    bb = _binary(b)
    inter = int(np.count_nonzero((aa > 0) & (bb > 0)))
    union = int(np.count_nonzero((aa > 0) | (bb > 0)))
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def mask_boundary(mask, radius=2):
    """Extract a thin binary boundary from a mask."""
    radius = max(1, int(radius))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    m = _binary(mask) * 255
    dil = cv2.dilate(m, k)
    ero = cv2.erode(m, k)
    return ((dil > 0) & (ero == 0)).astype(np.uint8) * 255


def boundary_f1(pred, target, tolerance=5):
    """Boundary F1 with a pixel tolerance."""
    pred_b = mask_boundary(pred)
    target_b = mask_boundary(target)
    pred_count = int(np.count_nonzero(pred_b))
    target_count = int(np.count_nonzero(target_b))
    if pred_count == 0 and target_count == 0:
        return 1.0
    if pred_count == 0 or target_count == 0:
        return 0.0

    tol = max(1, int(tolerance))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))
    target_band = cv2.dilate(target_b, k)
    pred_band = cv2.dilate(pred_b, k)

    precision = np.count_nonzero((pred_b > 0) & (target_band > 0)) / float(pred_count)
    recall = np.count_nonzero((target_b > 0) & (pred_band > 0)) / float(target_count)
    denom = precision + recall
    if denom <= 1e-8:
        return 0.0
    return float(2.0 * precision * recall / denom)


def chamfer_distance(pred, target, max_distance=80):
    """Symmetric boundary Chamfer distance in pixels, clipped for stable scoring."""
    pred_b = mask_boundary(pred)
    target_b = mask_boundary(target)
    pred_pts = np.count_nonzero(pred_b)
    target_pts = np.count_nonzero(target_b)
    if pred_pts == 0 and target_pts == 0:
        return 0.0
    if pred_pts == 0 or target_pts == 0:
        return float(max_distance)

    max_distance = float(max_distance)
    dt_target = cv2.distanceTransform((target_b == 0).astype(np.uint8), cv2.DIST_L2, 3)
    dt_pred = cv2.distanceTransform((pred_b == 0).astype(np.uint8), cv2.DIST_L2, 3)
    d1 = float(np.mean(dt_target[pred_b > 0]))
    d2 = float(np.mean(dt_pred[target_b > 0]))
    return float(min((d1 + d2) * 0.5, max_distance))


def alignment_score(pred, target, boundary_tolerance=6):
    """Compact score and metric dict for one projected-vs-image mask pair."""
    iou = mask_iou(pred, target)
    bf1 = boundary_f1(pred, target, tolerance=boundary_tolerance)
    chamfer = chamfer_distance(pred, target)
    score = iou + 0.55 * bf1 - 0.012 * chamfer
    return float(score), {"iou": iou, "boundary_f1": bf1, "chamfer": chamfer}
