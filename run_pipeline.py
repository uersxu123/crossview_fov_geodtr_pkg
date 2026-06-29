# -*- coding: utf-8 -*-
"""
一键运行：手机图 -> 遥感 patch 检索 -> 位姿优化 -> FOV 输出。
PyCharm：直接运行本文件。需要先改 config.py 里的图片路径和 GPS/YAW 初值。
"""
from pathlib import Path

import config as C
from src.io_utils import ensure_dir, imread, imwrite
from src.mask_projection import (
    draw_projected_masks_on_sat,
    projected_area_m2,
    project_phone_mask_to_sat,
)
from src.phone_annotation import annotate_phone_ground
from src.phone_refine import draw_phone_mask_overlay, refine_sat_projected_masks
from src.pose_local_opt import local_refine_pose_with_phone_mask
from src.pose_refine import draw_best_overlay, refine_pose, save_results_csv
from src.retrieval import draw_retrieval_heatmap, retrieve_topk
from src.sat_features import build_sat_features
from src.sat_to_phone import (
    draw_sat_guided_phone_overlay,
    project_sat_feature_masks_to_phone,
)


def main():
    ensure_dir(C.OUTPUT_DIR)
    phone = imread(C.PHONE_IMG_PATH)
    sat = imread(C.SAT_IMG_PATH)

    print("[1/4] satellite patch retrieval...")
    top, centers, scores = retrieve_topk(phone, sat)
    heatmap = draw_retrieval_heatmap(sat, centers, scores, top)
    imwrite(Path(C.OUTPUT_DIR) / "01_retrieval_heatmap.png", heatmap)
    print("Top retrieval candidates:")
    for i, t in enumerate(top[:10], start=1):
        print(f"  {i:02d}. x={t['x']:.1f}, y={t['y']:.1f}, score={t['score']:.4f}")

    print("[2/4] satellite feature extraction...")
    sat_feat = build_sat_features(sat)
    imwrite(Path(C.OUTPUT_DIR) / "03_satellite_features.png", sat_feat["debug"])

    print("[3/4] pose refinement...")
    results = refine_pose(phone, sat, sat_feat, top)
    best = results[0]

    print("[4/4] save overlay, masks and measurements...")
    overlay = draw_best_overlay(sat, phone, results)
    imwrite(Path(C.OUTPUT_DIR) / "02_best_fov_overlay.png", overlay)
    save_results_csv(Path(C.OUTPUT_DIR) / "04_top_candidates.csv", results)

    sat_to_phone_masks = project_sat_feature_masks_to_phone(sat_feat, best.pose, phone.shape)
    sat_to_phone_overlay = draw_sat_guided_phone_overlay(
        phone,
        sat_to_phone_masks["road_mask"],
        sat_to_phone_masks["green_mask"],
    )
    imwrite(Path(C.OUTPUT_DIR) / "05_sat_mask_to_phone_overlay.png", sat_to_phone_overlay)
    imwrite(Path(C.OUTPUT_DIR) / "06_sat_to_phone_road_mask.png", sat_to_phone_masks["road_mask"])
    imwrite(Path(C.OUTPUT_DIR) / "07_sat_to_phone_green_mask.png", sat_to_phone_masks["green_mask"])

    phone_anno, phone_masks = annotate_phone_ground(phone, sat_feat, best)
    imwrite(Path(C.OUTPUT_DIR) / "08_phone_ground_annotation_refined.png", phone_anno)
    imwrite(Path(C.OUTPUT_DIR) / "09_phone_road_mask_refined.png", phone_masks["road_mask"])
    imwrite(Path(C.OUTPUT_DIR) / "10_phone_green_mask_refined.png", phone_masks["green_mask"])

    print("[4.1/4] closed loop: phone-image target masks + local pose search...")
    grabcut_masks = refine_sat_projected_masks(
        phone,
        sat_to_phone_masks["road_mask"],
        sat_to_phone_masks["green_mask"],
    )
    grabcut_overlay = draw_phone_mask_overlay(
        phone,
        grabcut_masks["road_mask"],
        grabcut_masks["green_mask"],
    )
    imwrite(Path(C.OUTPUT_DIR) / "15_phone_grabcut_from_sat_overlay.png", grabcut_overlay)
    imwrite(Path(C.OUTPUT_DIR) / "16_phone_grabcut_road_mask.png", grabcut_masks["road_mask"])
    imwrite(Path(C.OUTPUT_DIR) / "17_phone_grabcut_green_mask.png", grabcut_masks["green_mask"])

    local_best, local_results = local_refine_pose_with_phone_mask(best, sat_feat, phone.shape, phone_masks)
    final_overlay, final_masks = annotate_phone_ground(phone, sat_feat, local_best)
    local_pose_overlay = draw_best_overlay(sat, phone, [local_best] + local_results[1:])
    imwrite(Path(C.OUTPUT_DIR) / "18_local_pose_fov_overlay.png", local_pose_overlay)
    imwrite(Path(C.OUTPUT_DIR) / "19_final_closed_loop_overlay.png", final_overlay)
    imwrite(Path(C.OUTPUT_DIR) / "20_final_closed_loop_road_mask.png", final_masks["road_mask"])
    imwrite(Path(C.OUTPUT_DIR) / "21_final_closed_loop_green_mask.png", final_masks["green_mask"])

    sat_road = project_phone_mask_to_sat(final_masks["road_mask"], local_best.pose, phone.shape, sat.shape)
    sat_green = project_phone_mask_to_sat(final_masks["green_mask"], local_best.pose, phone.shape, sat.shape)
    sat_green[sat_road > 0] = 0
    sat_mask_overlay = draw_projected_masks_on_sat(sat, sat_road, sat_green)
    imwrite(Path(C.OUTPUT_DIR) / "11_phone_to_sat_road_mask.png", sat_road)
    imwrite(Path(C.OUTPUT_DIR) / "12_phone_to_sat_green_mask.png", sat_green)
    imwrite(Path(C.OUTPUT_DIR) / "13_phone_to_sat_masks_overlay.png", sat_mask_overlay)

    measurement_csv = Path(C.OUTPUT_DIR) / "14_measurements.csv"
    measurement_csv.write_text(
        "name,pixels,area_m2,mpp\n"
        f"road,{int((sat_road > 0).sum())},{projected_area_m2(sat_road, local_best.pose):.4f},{local_best.pose.mpp:.6f}\n"
        f"green,{int((sat_green > 0).sum())},{projected_area_m2(sat_green, local_best.pose):.4f},{local_best.pose.mpp:.6f}\n",
        encoding="utf-8",
    )

    print("\n========== BEST ==========")
    print(f"score={best.score:.4f}")
    print(f"camera=({best.pose.x:.2f}, {best.pose.y:.2f})")
    print(f"yaw={best.pose.yaw:.2f}, pitch={best.pose.pitch:.2f}, hfov={best.pose.hfov:.2f}, mpp={best.pose.mpp:.3f}")
    print(f"gps_dist={best.parts['gps_dist']:.2f}, retrieval={best.retrieval_score:.4f}, road={best.parts['road']:.3f}")
    print("\n===== LOCAL POSE BEST =====")
    print(
        f"score={local_best.score:.4f}, "
        f"mask_score={local_best.parts.get('local_mask_score', 0.0):.4f}, "
        f"adjusted={local_best.parts.get('local_adjusted_mask_score', 0.0):.4f}, "
        f"delta_penalty={local_best.parts.get('local_delta_penalty', 0.0):.4f}"
    )
    print(f"camera=({local_best.pose.x:.2f}, {local_best.pose.y:.2f})")
    print(
        f"yaw={local_best.pose.yaw:.2f}, pitch={local_best.pose.pitch:.2f}, "
        f"hfov={local_best.pose.hfov:.2f}, mpp={local_best.pose.mpp:.3f}"
    )
    print(
        f"road_iou={local_best.parts.get('local_road_iou', 0.0):.3f}, "
        f"green_iou={local_best.parts.get('local_green_iou', 0.0):.3f}, "
        f"road_bf1={local_best.parts.get('local_road_boundary_f1', 0.0):.3f}, "
        f"green_bf1={local_best.parts.get('local_green_boundary_f1', 0.0):.3f}"
    )
    print("\n输出：")
    print(Path(C.OUTPUT_DIR) / "01_retrieval_heatmap.png")
    print(Path(C.OUTPUT_DIR) / "02_best_fov_overlay.png")
    print(Path(C.OUTPUT_DIR) / "03_satellite_features.png")
    print(Path(C.OUTPUT_DIR) / "04_top_candidates.csv")
    print(Path(C.OUTPUT_DIR) / "05_sat_mask_to_phone_overlay.png")
    print(Path(C.OUTPUT_DIR) / "08_phone_ground_annotation_refined.png")
    print(Path(C.OUTPUT_DIR) / "15_phone_grabcut_from_sat_overlay.png")
    print(Path(C.OUTPUT_DIR) / "18_local_pose_fov_overlay.png")
    print(Path(C.OUTPUT_DIR) / "19_final_closed_loop_overlay.png")
    print(Path(C.OUTPUT_DIR) / "13_phone_to_sat_masks_overlay.png")
    print(Path(C.OUTPUT_DIR) / "14_measurements.csv")


if __name__ == "__main__":
    main()
