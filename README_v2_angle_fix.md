# v2_angle_fix 说明

这版针对“真实方向应比当前输出更靠上方”的情况做了修改：

1. `config.py` 新增 `YAW_BIAS_DEG = 15.0`，会把 yaw 搜索中心从 235° 调到 250° 左右。
2. 新增 `LOCK_CAMERA_TO_GPS = True`，强制相机点从红圈/GPS 点发出，避免优化器把 camera 吸到左侧。
3. 收紧 `SEARCH_XY_RADIUS_PX` 和 `SEARCH_YAW_RADIUS_DEG`，减少跑偏。
4. 降低 road overlap 权重、提高 boundary edge 权重。你当前 road mask 太宽，`road=1.00` 已经没有角度区分能力。
5. `ROAD_LAB_DIST_THRESHOLD` 从 38 降到 30，避免道路 mask 泛滥。

如果还要更靠上：把 `YAW_BIAS_DEG` 从 `15.0` 改成 `18.0` 或 `22.0`。
如果偏得太上：改成 `10.0` 或 `6.0`。
