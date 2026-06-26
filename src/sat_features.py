# -*- coding: utf-8 -*-
import cv2
import numpy as np
import config as C
from .geometry import vector_from_bearing


def remove_red_annotation(img):
    if not C.REMOVE_RED_ANNOTATION:
        return img.copy()
    b, g, r = cv2.split(img)
    mask = ((r > 140) & (g < 135) & (b < 135)).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return img.copy()
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def build_sat_features(sat_img):
    h, w = sat_img.shape[:2]
    clean = remove_red_annotation(sat_img)
    hsv = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB).astype(np.float32)

    # local mask
    local = np.zeros((h, w), dtype=np.uint8)
    gx, gy = map(int, C.GPS_XY_INIT)
    cv2.circle(local, (gx, gy), C.SAT_FEATURE_RADIUS_PX, 255, -1)

    # road color seeds along initial yaw
    fwd = vector_from_bearing(C.YAW_INIT_DEG)
    seed_cols = []
    for d in C.ROAD_SEED_FORWARD_PX:
        p = np.array(C.GPS_XY_INIT, dtype=float) + fwd * d
        x = int(np.clip(round(p[0]), 0, w - 1))
        y = int(np.clip(round(p[1]), 0, h - 1))
        patch = lab[max(0,y-2):min(h,y+3), max(0,x-2):min(w,x+3)].reshape(-1, 3)
        if len(patch):
            seed_cols.append(np.mean(patch, axis=0))
    if not seed_cols:
        seed_cols = [np.mean(lab.reshape(-1,3), axis=0)]

    dist_min = np.full((h, w), 1e9, dtype=np.float32)
    for c in seed_cols:
        d = np.linalg.norm(lab - c.reshape(1, 1, 3), axis=2)
        dist_min = np.minimum(dist_min, d)
    road = (dist_min < C.ROAD_LAB_DIST_THRESHOLD).astype(np.uint8) * 255
    road = cv2.bitwise_and(road, local)

    H, S, V = cv2.split(hsv)
    green = (((H >= 35) & (H <= 95) & (S > 35) & (V > 40))).astype(np.uint8) * 255
    green = cv2.bitwise_and(green, local)
    water = (((H >= 70) & (H <= 125) & (S > 20) & (V < 135))).astype(np.uint8) * 255
    bad = cv2.bitwise_or(green, water)
    bad = cv2.bitwise_and(bad, local)
    road[bad > 0] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    road = cv2.morphologyEx(road, cv2.MORPH_OPEN, k)
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, k)

    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edge = cv2.Canny(gray, 50, 150)
    edge = cv2.bitwise_and(edge, local)

    road_dist = cv2.distanceTransform(255 - road, cv2.DIST_L2, 5)
    road_score = np.exp(-road_dist / 18.0).astype(np.float32)
    edge_dist = cv2.distanceTransform(255 - edge, cv2.DIST_L2, 5)
    edge_score = np.exp(-edge_dist / 10.0).astype(np.float32)
    bad_dist = cv2.distanceTransform(255 - bad, cv2.DIST_L2, 5)
    bad_score = np.maximum((bad > 0).astype(np.float32), np.exp(-bad_dist / 10.0).astype(np.float32) * 0.45)

    debug = sat_img.copy()
    road_color = np.zeros_like(debug); road_color[:,:,1] = road
    bad_color = np.zeros_like(debug); bad_color[:,:,2] = bad
    debug = cv2.addWeighted(debug, 0.75, road_color, 0.25, 0)
    debug = cv2.addWeighted(debug, 0.82, bad_color, 0.25, 0)
    cv2.circle(debug, (gx, gy), 10, (255,0,255), 2)
    return {"road_score": road_score, "edge_score": edge_score, "bad_score": bad_score, "road_mask": road, "green_mask": green, "bad_mask": bad, "edge": edge, "debug": debug}
