# -*- coding: utf-8 -*-
"""
PyCharm 直接改这个文件即可。
"""

# =========================
# 1. 输入输出
# =========================
PHONE_IMG_PATH = r"data/phone1.jpg"
SAT_IMG_PATH = r"data/satellite1.png"
OUTPUT_DIR = r"output"

# =========================
# 2. 粗初值：手机/GPS 给的值可以不准
# =========================
GPS_XY_INIT = (230, 498)       # 遥感截图像素坐标，不是经纬度
YAW_INIT_DEG = 179             # 0=上，90=右，180=下，270=左
# 人工/经验微调：在 YAW_INIT_DEG 基础上整体平移搜索中心。
# 当前以 179 为主方向，先保持 0；只有 FOV 整体系统性偏转时再小幅调整。
YAW_BIAS_DEG = 0.0

# 是否强制相机点固定在 GPS_XY_INIT。
# 你这张图建议 True，避免优化器把 camera 从红圈吸到左侧候选点。
LOCK_CAMERA_TO_GPS = True
HFOV_INIT_DEG = 70.0
PITCH_INIT_DEG = 5.0              # 向下为正
METERS_PER_PIXEL_INIT = 0.50
CAMERA_HEIGHT_M = 1.60

# =========================
# 3. 特征后端
# =========================
# "opencv"：不需要模型权重，先跑通流程
# "geodtr"：需要你下载 GeoDTR+ 官方权重，并补齐 src/feature_backend.py 的模型加载接口
FEATURE_BACKEND = "geodtr"
GEODTR_REPO_DIR = r"external/geodtr_plus"
GEODTR_WEIGHT_DIR = r"weights/geodtr_plus/VIGOR_cross"
GEODTR_ADAPTER_PATH = r"adapters/geodtr_plus_vigor_adapter.py"
GEODTR_INPUT_SIZE = 384
DEVICE = "cuda"  # cuda / cpu

# =========================
# 4. 遥感 patch 检索参数
# =========================
PATCH_SIZE = 256
PATCH_STRIDE = 32
PATCH_SEARCH_RADIUS = 260
TOPK_PATCHES = 25

# =========================
# 5. 位姿优化搜索范围
# =========================
SEARCH_XY_RADIUS_PX = 35.0
SEARCH_YAW_RADIUS_DEG = 10.0
PITCH_MIN_DEG = 1.0
PITCH_MAX_DEG = 8.0
HFOV_MIN_DEG = 50.0
HFOV_MAX_DEG = 72.0
MPP_MIN = 0.35
MPP_MAX = 0.72
MAX_RANGE_M = 220.0
NEAR_RANGE_M = 1.0

# =========================
# 6. 手机图地面区域
# =========================
PHONE_GROUND_TOP_RATIO = 0.43
PHONE_GROUND_BOTTOM_RATIO = 0.98
PHONE_SIDE_MARGIN_RATIO = 0.04

# =========================
# 7. 优化器参数
# =========================
RANDOM_SEED = 42
CANDIDATES_PER_PATCH = 220
LOCAL_REFINE_ROUNDS = 3
SAVE_TOPK_POSES = 80

# =========================
# 8. 评分权重
# =========================
W_DEEP_RETRIEVAL = 0.28
# 你的 road mask 已经几乎全绿，road=1.00，不能给太高权重，否则区分不了角度。
W_ROAD_OVERLAP = 0.55
W_CENTERLINE_ROAD = 0.80
# 加强边界/结构贴合，让角度由道路边界和建筑/路缘线决定。
W_BOUNDARY_EDGE = 1.20
W_BAD_REGION = 1.10
W_GPS_PRIOR = 0.55
W_YAW_PRIOR = 1.20
W_CAMERA_PRIOR = 0.55
W_SHAPE = 0.70

# =========================
# 9. 遥感图道路/惩罚区提取
# =========================
ROAD_LAB_DIST_THRESHOLD = 30.0
REMOVE_RED_ANNOTATION = True
SAT_FEATURE_RADIUS_PX = 420
ROAD_SEED_FORWARD_PX = [20, 45, 75, 110, 150]

# =========================
# 10. 手机图道路/绿化带标注
# =========================
# "map_guided_grabcut": 使用地图投影先验 + 手机图颜色种子 + GrabCut 做道路/绿化带分割。
# "reference_contour": 输出接近人工示例的粗线轮廓标注，仅用于调试对照。
# "auto_mask": 旧版自动分割的半透明覆盖标注。
PHONE_ANNOTATION_STYLE = "map_guided_grabcut"
PHONE_SEG_MAX_DIM = 1400
PHONE_SEG_GRABCUT_ITERS = 3
PHONE_SEG_VIS_ALPHA = 0.42

# =========================
# 11. 手机图语义分割与投影
# =========================
PHONE_SEMSEG_ENABLED = True
PHONE_SEMSEG_STRICT = False
PHONE_SEMSEG_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
PHONE_SEMSEG_MAX_DIM = 1536
PHONE_SEMSEG_ROAD_LABELS = ("road",)
PHONE_SEMSEG_GREEN_LABELS = ("vegetation", "terrain")
PHONE_SEMSEG_IGNORE_LABELS = (
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
)
PHONE_SEMSEG_GREEN_TOP_RATIO = 0.50
PHONE_SEMSEG_ROAD_CLOSE_K = 45
PHONE_SEMSEG_GREEN_CLOSE_K = 35
PHONE_TO_SAT_SAMPLE_STEP = 2
PHONE_TO_SAT_DILATE_PX = 4
PHONE_TO_SAT_CLOSE_W = 9
PHONE_TO_SAT_CLOSE_H = 23

# =========================
# 12. Satellite mask -> phone mask
# =========================
# Sample every N phone pixels when projecting to satellite masks. 1 is most accurate but slower.
SAT_TO_PHONE_SAMPLE_STEP = 2
# Morphology cleanup for satellite-guided masks on the phone image.
SAT_TO_PHONE_CLOSE_K = 15
SAT_TO_PHONE_OPEN_K = 5
SAT_TO_PHONE_MIN_AREA = 800
# Keep only likely ground/lower-image pixels to reduce sky/building false positives.
SAT_TO_PHONE_GROUND_ONLY = True
SAT_TO_PHONE_VIS_ALPHA = 0.42

# =========================
# 13. 闭环优化：粗投影调试 + 手机图 target mask + 局部 pose 搜索
# =========================
# 不使用 SAM；15/16/17 是卫星粗投影引导的 GrabCut 调试结果。
# 18/19/20/21 使用 08/09/10 的手机图 refined mask 作为 pose 校正目标，更稳。
PHONE_REFINE_MAX_DIM = 1200
PHONE_REFINE_GRABCUT_ITERS = 4
PHONE_REFINE_SURE_FG_ERODE_K = 17
PHONE_REFINE_PROBABLE_FG_DILATE_K = 35
PHONE_REFINE_SURE_BG_DILATE_K = 75
PHONE_REFINE_CLOSE_K = 17
PHONE_REFINE_OPEN_K = 5
PHONE_REFINE_FINAL_CLOSE_K = 9
PHONE_REFINE_FINAL_OPEN_K = 3
PHONE_REFINE_MIN_AREA = 900
PHONE_REFINE_VIS_ALPHA = 0.42

# 在 best.pose 周围小范围搜索，让重新投影的 mask 更贴合手机图 refined mask。
POSE_LOCAL_MAX_DIM = 900
POSE_LOCAL_SAMPLE_STEP = 5
POSE_LOCAL_BOUNDARY_TOLERANCE = 6
POSE_LOCAL_CANDIDATES_PER_ROUND = 70
POSE_LOCAL_KEEP_PARENTS = 4
POSE_LOCAL_SAVE_TOPK = 20
POSE_LOCAL_SCORE_WEIGHT = 0.35
POSE_LOCAL_RECALL_WEIGHT = 0.35
POSE_LOCAL_AREA_WEIGHT = 0.25
POSE_LOCAL_DELTA_PENALTY_WEIGHT = 0.18
POSE_LOCAL_RANDOM_SEED = 42
POSE_LOCAL_XY_RADIUS_PX = 15.0
POSE_LOCAL_YAW_RADIUS_DEG = 6.0
POSE_LOCAL_PITCH_RADIUS_DEG = 2.5
POSE_LOCAL_HFOV_RADIUS_DEG = 3.0
POSE_LOCAL_MPP_RADIUS = 0.025
