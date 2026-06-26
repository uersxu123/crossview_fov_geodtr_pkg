# CrossView FOV GeoDTR Package

这是一个“手机图 → 遥感图 → 视场角落图”的独立工程骨架，适合在 PyCharm 里直接运行。

## 这包能做什么

1. 以 GPS 粗位置为中心，在遥感图附近裁剪 patch。
2. 用特征检索找到最可能的 satellite patch 候选。
3. 对 top-k 候选做几何位姿优化：`x, y, yaw, pitch, hfov, meters_per_pixel`。
4. 输出最终手机视场角 polygon、候选热力图、top-k 参数表。

## 训练好的模型说明

GeoDTR+ 官方仓库提供预训练权重，但权重不适合直接随第三方包重新分发。你需要自己从官方仓库 README 里的 Google Drive 下载。

下载后建议放到：

```text
weights/geodtr_plus/
    CVUSA/
    CVACT/
    CVUSA_NP/
    CVACT_NP/
    VIGOR_same/
    VIGOR_cross/
```

本包默认 `FEATURE_BACKEND = "opencv"`，无需权重也能跑通流程；真正使用 GeoDTR+ 时，把 `config.py` 里的：

```python
FEATURE_BACKEND = "geodtr"
GEODTR_REPO_DIR = r"D:\path\to\GeoDTR_plus"
GEODTR_WEIGHT_DIR = r"D:\path\to\weights\geodtr_plus\VIGOR_cross"
```

并在 `src/feature_backend.py` 里按你的 GeoDTR+ 模型权重格式补齐 `GeoDTRFeatureBackend` 的 load_model 部分。

> 说明：GeoDTR+ 各版本/权重文件命名不完全固定，直接硬编码可能会和你的下载版本不一致，所以这里保留了清晰接口。

## 快速运行

```bash
pip install -r requirements.txt
python run_pipeline.py
```

你只需要改 `config.py` 顶部：

```python
PHONE_IMG_PATH = r"data/phone.jpg"
SAT_IMG_PATH = r"data/satellite.png"
GPS_XY_INIT = (858.0, 80.0)
YAW_INIT_DEG = 235.0
```

## 输出文件

```text
output/01_retrieval_heatmap.png
output/02_best_fov_overlay.png
output/03_satellite_features.png
output/04_top_candidates.csv
```

## 推荐使用方式

第一阶段先用 OpenCV fallback 跑通完整流程；第二阶段接 GeoDTR+ 预训练模型做 top-k 召回；第三阶段如果 top-k 能覆盖正确区域，再微调 GeoDTR+ 或增加你自己的校园样本。
