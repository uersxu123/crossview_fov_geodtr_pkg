# -*- coding: utf-8 -*-
import numpy as np
import cv2
import config as C
from .patches import generate_satellite_patches, phone_to_query
from .feature_backend import get_backend, cosine_scores


def retrieve_topk(phone_img, sat_img):
    query = phone_to_query(phone_img)
    centers, patches = generate_satellite_patches(sat_img)
    backend = get_backend()
    if hasattr(backend, "encode_query"):
        q_feat = backend.encode_query([query])[0]
    else:
        q_feat = backend.encode_many([query])[0]
    if hasattr(backend, "encode_gallery"):
        g_feat = backend.encode_gallery(patches)
    else:
        g_feat = backend.encode_many(patches)
    scores = cosine_scores(q_feat, g_feat)
    order = np.argsort(-scores)
    top = []
    for idx in order[:C.TOPK_PATCHES]:
        x, y = centers[idx]
        top.append({"x": x, "y": y, "score": float(scores[idx]), "patch_index": int(idx)})
    return top, centers, scores


def draw_retrieval_heatmap(sat_img, centers, scores, top):
    out = sat_img.copy()
    h, w = out.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    if len(scores) == 0:
        return out
    s = np.asarray(scores, dtype=np.float32)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    for (x, y), v in zip(centers, s):
        cv2.circle(heat, (int(round(x)), int(round(y))), max(3, C.PATCH_STRIDE // 2), float(v), -1)
    heat_u8 = np.uint8(np.clip(heat * 255, 0, 255))
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    mask = heat_u8 > 0
    out[mask] = cv2.addWeighted(out, 0.55, color, 0.45, 0)[mask]
    for i, t in enumerate(top[:10], start=1):
        x, y = int(t["x"]), int(t["y"])
        cv2.circle(out, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(out, str(i), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    gx, gy = map(int, C.GPS_XY_INIT)
    cv2.circle(out, (gx, gy), 10, (255, 0, 255), 2)
    cv2.putText(out, "GPS", (gx + 10, gy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
    return out
