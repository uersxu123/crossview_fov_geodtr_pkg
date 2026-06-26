# 如何接入 GeoDTR+ / GeoGTR

当前工程默认使用：

```python
FEATURE_BACKEND = "opencv"
```

这能跑通完整流程，但只是 OpenCV fallback，不是真正的 GeoDTR+/GeoGTR 深度检索模型。

## 你需要准备的文件

要真正使用 GeoDTR+/GeoGTR，需要额外准备：

1. 官方 GeoDTR+/GeoGTR 代码仓库
2. 官方预训练权重
3. 一个适配器文件，用来告诉本工程如何构建模型、加载权重、提取特征

本工程已经提供模板：

```text
adapters/geodtr_plus_adapter_template.py
```

## 配置方式

在 `config.py` 中设置：

```python
FEATURE_BACKEND = "geodtr"
GEODTR_REPO_DIR = r"D:\path\to\GeoDTR_plus"
GEODTR_WEIGHT_DIR = r"D:\path\to\weights\geodtr_plus\VIGOR_cross"
GEODTR_ADAPTER_PATH = r"adapters\my_geodtr_adapter.py"
GEODTR_INPUT_SIZE = 384
```

如果你的导师说的是 `GeoGTR`，也可以写：

```python
FEATURE_BACKEND = "geogtr"
```

工程内部会走同一套外部模型适配器机制。

## 适配器需要实现什么

适配器必须提供以下二选一：

```python
def build_backend(config):
    ...
```

或：

```python
class GeoDTRAdapter:
    ...
```

返回对象必须实现：

```python
encode_many(images_bgr) -> np.ndarray
```

返回值形状必须是：

```text
[N, D]
```

也就是 N 张图片对应 N 个 D 维特征向量。

## 为什么不能现在直接用

当前项目里没有官方 GeoDTR+/GeoGTR 仓库，也没有官方 checkpoint 权重。`weights/` 目录目前只有说明文件，所以不能凭空跑出真正的深度特征。

拿到官方仓库和权重后，把官方 `test.py` / `test_vigor.py` 里的三部分代码搬到 adapter 里：

1. model 构建
2. checkpoint 加载
3. feature extraction

之后运行：

```bash
python -B run_pipeline.py
```

主流程不需要改。
