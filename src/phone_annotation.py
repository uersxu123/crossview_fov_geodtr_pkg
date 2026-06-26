# -*- coding: utf-8 -*-
import cv2
import numpy as np
import config as C
from .geometry import project_uv_to_sat
from .semantic_segmentation import try_segment_phone


def _sample_mask(mask, pts):
    h, w = mask.shape[:2]
    xs = np.rint(pts[:, 0]).astype(np.int32)
    ys = np.rint(pts[:, 1]).astype(np.int32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    out = np.zeros(len(pts), dtype=bool)
    out[valid] = mask[ys[valid], xs[valid]] > 0
    return out


def _phone_ground_candidate(phone_img):
    h, w = phone_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    top = int(h * max(C.PHONE_GROUND_TOP_RATIO, 0.0))
    bottom = int(h * C.PHONE_GROUND_BOTTOM_RATIO)
    left = int(w * C.PHONE_SIDE_MARGIN_RATIO)
    right = int(w * (1.0 - C.PHONE_SIDE_MARGIN_RATIO))
    mask[top:bottom, left:right] = 255

    hsv = cv2.cvtColor(phone_img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    sky = ((S < 45) & (V > 145) & (np.indices((h, w))[0] < h * 0.62)).astype(np.uint8) * 255
    mask[sky > 0] = 0
    return mask


def _perspective_roi(shape, kind):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if kind == "road":
        pts = np.array([
            [int(w * 0.03), int(h * 0.98)],
            [int(w * 0.97), int(h * 0.98)],
            [int(w * 0.93), int(h * 0.70)],
            [int(w * 0.70), int(h * 0.49)],
            [int(w * 0.55), int(h * 0.43)],
            [int(w * 0.45), int(h * 0.43)],
            [int(w * 0.30), int(h * 0.49)],
            [int(w * 0.07), int(h * 0.70)],
        ], dtype=np.int32)
    else:
        pts = np.array([
            [0, int(h * 0.88)],
            [0, int(h * 0.55)],
            [int(w * 0.30), int(h * 0.45)],
            [int(w * 0.48), int(h * 0.42)],
            [int(w * 0.52), int(h * 0.42)],
            [int(w * 0.70), int(h * 0.45)],
            [w - 1, int(h * 0.55)],
            [w - 1, int(h * 0.88)],
            [int(w * 0.86), int(h * 0.98)],
            [int(w * 0.14), int(h * 0.98)],
        ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _phone_color_masks(phone_img):
    hsv = cv2.cvtColor(phone_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(phone_img, cv2.COLOR_BGR2LAB)
    H, S, V = cv2.split(hsv)
    L, A, B = cv2.split(lab)

    green = ((H >= 32) & (H <= 95) & (S > 28) & (V > 35)).astype(np.uint8) * 255
    yellow_or_orange = ((H >= 8) & (H <= 36) & (S > 80) & (V > 80))
    road = ((S < 95) & (V > 25) & (L > 25) & (A >= 105) & (A <= 155) & ~yellow_or_orange).astype(np.uint8) * 255
    return road, green


def _component_from_seed(mask, seed_rect):
    x0, y0, x1, y1 = seed_rect
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(mask.shape[1], x1); y1 = min(mask.shape[0], y1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask

    seed_labels = labels[y0:y1, x0:x1]
    ids, counts = np.unique(seed_labels[seed_labels > 0], return_counts=True)
    if len(ids):
        best = int(ids[np.argmax(counts)])
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas) + 1)
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def _clean(mask, min_area=800):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def _fill_nearby_gaps(mask, ksize=31):
    ksize = max(3, int(ksize) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _fill_internal_holes(mask):
    h, w = mask.shape[:2]
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def _points_from_ratio(shape, pts):
    h, w = shape[:2]
    return np.array([(int(round(x * (w - 1))), int(round(y * (h - 1)))) for x, y in pts], dtype=np.int32)


def _draw_reference_style_annotation(phone_img):
    h, w = phone_img.shape[:2]
    out = phone_img.copy()
    road_mask = np.zeros((h, w), dtype=np.uint8)
    green_mask = np.zeros((h, w), dtype=np.uint8)

    road_polys = [
        [
            (0.00, 0.99), (0.00, 0.79), (0.04, 0.74), (0.13, 0.69),
            (0.24, 0.64), (0.36, 0.59), (0.47, 0.52), (0.47, 0.50),
            (0.49, 0.48), (0.61, 0.48), (0.62, 0.52), (0.67, 0.58),
            (0.78, 0.64), (0.90, 0.72), (0.99, 0.80), (0.99, 0.99),
        ],
        [
            (0.00, 0.56), (0.12, 0.55), (0.25, 0.52), (0.40, 0.47),
            (0.50, 0.44), (0.58, 0.44),
        ],
        [
            (0.62, 0.55), (0.70, 0.58), (0.79, 0.63), (0.90, 0.71),
            (0.99, 0.77),
        ],
    ]
    green_polys = [
        [
            (0.00, 0.50), (0.14, 0.50), (0.30, 0.47), (0.45, 0.44),
            (0.50, 0.44), (0.47, 0.47), (0.34, 0.50), (0.17, 0.54),
            (0.00, 0.56),
        ],
        [
            (0.61, 0.47), (0.68, 0.47), (0.76, 0.50), (0.87, 0.54),
            (0.99, 0.58), (0.99, 0.75), (0.91, 0.70), (0.80, 0.63),
            (0.69, 0.57), (0.61, 0.52),
        ],
    ]

    for pts in road_polys[:1]:
        cv2.fillPoly(road_mask, [_points_from_ratio(phone_img.shape, pts)], 255)
    for pts in green_polys:
        cv2.fillPoly(green_mask, [_points_from_ratio(phone_img.shape, pts)], 255)
    green_mask[road_mask > 0] = 0

    stroke = max(12, int(min(h, w) * 0.011))
    road_color = (255, 120, 0)
    green_color = (0, 220, 255)

    for pts in green_polys:
        poly = _points_from_ratio(phone_img.shape, pts)
        cv2.polylines(out, [poly], True, green_color, stroke, cv2.LINE_AA)
    for i, pts in enumerate(road_polys):
        poly = _points_from_ratio(phone_img.shape, pts)
        cv2.polylines(out, [poly], i == 0, road_color, stroke, cv2.LINE_AA)

    return out, {"road_mask": road_mask, "green_mask": green_mask}


def _resize_for_segmentation(img):
    max_dim = int(getattr(C, "PHONE_SEG_MAX_DIM", 1400))
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale >= 1.0:
        return img.copy(), 1.0
    small = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return small, scale


def _resize_mask(mask, shape):
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def _run_grabcut(img, probable_fg, sure_fg, sure_bg, iters):
    mask = np.full(img.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    mask[probable_fg > 0] = cv2.GC_PR_FGD
    mask[sure_fg > 0] = cv2.GC_FGD
    mask[sure_bg > 0] = cv2.GC_BGD

    fg_count = int(np.count_nonzero((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)))
    bg_count = int(np.count_nonzero(mask == cv2.GC_BGD))
    if fg_count < 20 or bg_count < 20:
        return (probable_fg > 0).astype(np.uint8) * 255

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, None, bgd, fgd, int(iters), cv2.GC_INIT_WITH_MASK)
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return out


def _map_guided_segmentation(phone_img, sat_feat, best_result, sample_step=6):
    pose = best_result.pose
    h, w = phone_img.shape[:2]

    ys, xs = np.mgrid[0:h:sample_step, 0:w:sample_step]
    uv = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    sat_pts = project_uv_to_sat(uv, pose, phone_img.shape, sat_feat["road_mask"].shape)
    road_small = _sample_mask(sat_feat["road_mask"], sat_pts).reshape(ys.shape).astype(np.uint8) * 255
    green_small = _sample_mask(sat_feat["green_mask"], sat_pts).reshape(ys.shape).astype(np.uint8) * 255
    road_from_sat = cv2.resize(road_small, (w, h), interpolation=cv2.INTER_NEAREST)
    green_from_sat = cv2.resize(green_small, (w, h), interpolation=cv2.INTER_NEAREST)

    seg_img, scale = _resize_for_segmentation(phone_img)
    sh, sw = seg_img.shape[:2]
    ground = _resize_mask(_phone_ground_candidate(phone_img), seg_img.shape)
    road_roi = _resize_mask(_perspective_roi(phone_img.shape, "road"), seg_img.shape)
    green_roi = _resize_mask(_perspective_roi(phone_img.shape, "green"), seg_img.shape)
    road_sat = _resize_mask(road_from_sat, seg_img.shape)
    green_sat = _resize_mask(green_from_sat, seg_img.shape)
    road_color, green_color = _phone_color_masks(seg_img)
    semseg = try_segment_phone(phone_img)
    if semseg is not None:
        sem_road = _resize_mask(semseg["road_mask"], seg_img.shape)
        sem_green = _resize_mask(semseg["green_mask"], seg_img.shape)
        sem_ignore = _resize_mask(semseg["ignore_mask"], seg_img.shape)
    else:
        sem_road = np.zeros((sh, sw), dtype=np.uint8)
        sem_green = np.zeros((sh, sw), dtype=np.uint8)
        sem_ignore = np.zeros((sh, sw), dtype=np.uint8)

    hsv = cv2.cvtColor(seg_img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    yy = np.indices((sh, sw))[0]
    sky_or_high = (((S < 55) & (V > 135) & (yy < sh * 0.55)) | (yy < sh * 0.38)).astype(np.uint8) * 255

    road_prob = cv2.bitwise_and(cv2.bitwise_or(road_color, sem_road), ground)
    road_prob = cv2.bitwise_and(road_prob, road_roi)
    road_prob = cv2.bitwise_or(road_prob, cv2.bitwise_and(road_sat, road_roi))
    road_prob[sem_green > 0] = 0

    sure_road = np.zeros((sh, sw), dtype=np.uint8)
    sure_road[int(sh * 0.80):int(sh * 0.98), int(sw * 0.25):int(sw * 0.78)] = 255
    sure_road = cv2.bitwise_and(sure_road, cv2.bitwise_or(road_color, sem_road))
    if np.count_nonzero(sure_road) < 100:
        sure_road[int(sh * 0.76):int(sh * 0.98), int(sw * 0.18):int(sw * 0.88)] = 255
        sure_road = cv2.bitwise_and(sure_road, road_roi)

    road_bg = cv2.bitwise_or(sky_or_high, cv2.bitwise_or(sem_green, green_color))
    road_bg = cv2.bitwise_or(road_bg, cv2.bitwise_and(sem_ignore, cv2.bitwise_not(road_roi)))
    road_bg = cv2.bitwise_or(road_bg, cv2.bitwise_not(ground))
    road = _run_grabcut(
        seg_img,
        cv2.bitwise_or(road_prob, sure_road),
        sure_road,
        road_bg,
        getattr(C, "PHONE_SEG_GRABCUT_ITERS", 3),
    )
    road = cv2.bitwise_and(road, road_roi)
    # Keep the visible pavement connected through the perspective road corridor.
    # This recovers wet/reflective road regions that GrabCut may split.
    if semseg is not None:
        road = cv2.bitwise_or(road, cv2.bitwise_and(sem_road, cv2.bitwise_and(ground, road_roi)))
    road = cv2.bitwise_or(road, cv2.bitwise_and(road_prob, road_roi))
    road = _fill_nearby_gaps(road, getattr(C, "PHONE_SEMSEG_ROAD_CLOSE_K", 45))
    road = _component_from_seed(road, (int(sw * 0.18), int(sh * 0.72), int(sw * 0.85), int(sh * 0.99)))
    road = _fill_internal_holes(road)
    road = _clean(road, min_area=max(500, (sh * sw) // 7000))

    near_road = cv2.dilate(road, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
    green_top_ratio = float(getattr(C, "PHONE_SEMSEG_GREEN_TOP_RATIO", 0.50))
    low_green = np.zeros_like(green_color)
    low_green[(yy > sh * green_top_ratio) & ((green_color > 0) | (sem_green > 0))] = 255
    green_prob = cv2.bitwise_and(low_green, green_roi)
    green_prob = cv2.bitwise_and(green_prob, cv2.bitwise_or(near_road, green_sat))
    # Map projection is only a prior; require phone-image green pixels so
    # non-vegetation foreground objects are not absorbed into the green mask.
    green_prob = cv2.bitwise_or(green_prob, cv2.bitwise_and(cv2.bitwise_and(green_sat, green_roi), low_green))

    sure_green = cv2.erode(green_prob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    green_bg = cv2.bitwise_or(sky_or_high, road)
    green_bg = cv2.bitwise_or(green_bg, cv2.bitwise_not(green_roi))
    green = _run_grabcut(
        seg_img,
        green_prob,
        sure_green,
        green_bg,
        getattr(C, "PHONE_SEG_GRABCUT_ITERS", 3),
    )
    green = cv2.bitwise_and(green, green_roi)
    green[road > 0] = 0
    if semseg is not None:
        sem_green_low = cv2.bitwise_and(sem_green, green_roi)
        sem_green_low[yy <= sh * green_top_ratio] = 0
        sem_green_low = cv2.bitwise_and(sem_green_low, cv2.bitwise_or(near_road, green_sat))
        green = cv2.bitwise_or(green, sem_green_low)
        green[road > 0] = 0
    green = _fill_nearby_gaps(green, getattr(C, "PHONE_SEMSEG_GREEN_CLOSE_K", 35))
    green = _clean(green, min_area=max(400, (sh * sw) // 9000))
    green[road > 0] = 0

    road_full = _resize_mask(road, phone_img.shape)
    green_full = _resize_mask(green, phone_img.shape)
    green_full[road_full > 0] = 0
    return road_full, green_full


def annotate_phone_ground(phone_img, sat_feat, best_result, sample_step=4):
    style = getattr(C, "PHONE_ANNOTATION_STYLE", "map_guided_grabcut")
    if style == "reference_contour":
        return _draw_reference_style_annotation(phone_img)
    if style == "map_guided_grabcut":
        road, green = _map_guided_segmentation(phone_img, sat_feat, best_result, sample_step=sample_step)

        road_color_img = np.zeros_like(phone_img)
        road_color_img[:, :, 0] = 255
        green_color_img = np.zeros_like(phone_img)
        green_color_img[:, :, 1] = 230
        green_color_img[:, :, 2] = 255

        out = phone_img.copy()
        alpha = float(getattr(C, "PHONE_SEG_VIS_ALPHA", 0.42))
        road_blend = cv2.addWeighted(out, 1.0 - alpha, road_color_img, alpha, 0)
        green_blend = cv2.addWeighted(out, 1.0 - alpha, green_color_img, alpha, 0)
        out[road > 0] = road_blend[road > 0]
        out[green > 0] = green_blend[green > 0]
        contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 80, 0), 3)
        contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 200, 255), 3)
        return out, {"road_mask": road, "green_mask": green}

    pose = best_result.pose
    h, w = phone_img.shape[:2]
    ys, xs = np.mgrid[0:h:sample_step, 0:w:sample_step]
    uv = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    sat_pts = project_uv_to_sat(uv, pose, phone_img.shape, sat_feat["road_mask"].shape)

    road_small = _sample_mask(sat_feat["road_mask"], sat_pts).reshape(ys.shape).astype(np.uint8) * 255
    green_small = _sample_mask(sat_feat["green_mask"], sat_pts).reshape(ys.shape).astype(np.uint8) * 255
    road_from_sat = cv2.resize(road_small, (w, h), interpolation=cv2.INTER_NEAREST)
    green_from_sat = cv2.resize(green_small, (w, h), interpolation=cv2.INTER_NEAREST)

    ground = _phone_ground_candidate(phone_img)
    road_color, green_color = _phone_color_masks(phone_img)

    road_roi = _perspective_roi(phone_img.shape, "road")
    green_roi = _perspective_roi(phone_img.shape, "green")

    road_candidate = cv2.bitwise_and(road_color, ground)
    road_candidate = cv2.bitwise_and(road_candidate, road_roi)
    road_candidate[green_color > 0] = 0
    # Satellite projection can recover weak road pixels, but only inside the road ROI.
    sat_road_hint = cv2.bitwise_and(road_from_sat, road_roi)
    sat_road_hint = cv2.bitwise_and(sat_road_hint, ground)
    road_candidate = cv2.bitwise_or(road_candidate, cv2.bitwise_and(sat_road_hint, road_color))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    road_candidate = cv2.morphologyEx(road_candidate, cv2.MORPH_CLOSE, k)
    road = _component_from_seed(
        road_candidate,
        (int(w * 0.25), int(h * 0.78), int(w * 0.75), int(h * 0.99)),
    )
    road = _clean(road, min_area=max(2000, (h * w) // 8000))

    yy = np.indices((h, w))[0]
    near_road = cv2.dilate(road, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121)))
    low_green = np.zeros_like(green_color)
    low_green[(yy > h * 0.43) & (green_color > 0)] = 255
    green = cv2.bitwise_and(low_green, ground)
    green = cv2.bitwise_and(green, green_roi)
    green = cv2.bitwise_and(green, near_road)
    sat_green_hint = cv2.bitwise_and(green_from_sat, low_green)
    sat_green_hint = cv2.bitwise_and(sat_green_hint, green_roi)
    green = cv2.bitwise_or(green, sat_green_hint)
    green[road > 0] = 0
    green = _clean(green, min_area=max(1000, (h * w) // 12000))
    green[road > 0] = 0

    overlay = phone_img.copy()
    road_color_img = np.zeros_like(phone_img)
    road_color_img[:, :, 0] = 255
    green_color_img = np.zeros_like(phone_img)
    green_color_img[:, :, 1] = 230
    green_color_img[:, :, 2] = 255

    out = overlay.copy()
    out[road > 0] = cv2.addWeighted(overlay, 0.55, road_color_img, 0.45, 0)[road > 0]
    out[green > 0] = cv2.addWeighted(out, 0.55, green_color_img, 0.45, 0)[green > 0]

    contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 80, 0), 3)
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 200, 255), 3)
    return out, {"road_mask": road, "green_mask": green}
