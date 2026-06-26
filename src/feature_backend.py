# -*- coding: utf-8 -*-
import importlib.util
import os
import sys

import cv2
import numpy as np

import config as C


class OpenCVFeatureBackend:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1200)

    def _hist_feature(self, img):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def _edge_feature(self, img):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 60, 160)
        small = cv2.resize(edge, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return small.flatten()

    def encode_one(self, img):
        feat = np.concatenate([self._hist_feature(img), self._edge_feature(img)], axis=0)
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        return feat.astype(np.float32)

    def encode_many(self, imgs):
        return np.stack([self.encode_one(x) for x in imgs], axis=0)


class ExternalGeoDTRFeatureBackend:
    """Adapter-based backend for GeoDTR+/GeoGTR style repositories.

    The official projects differ in model class names, checkpoint keys, and
    test script arguments. To avoid hard-coding one version here, this backend
    loads a small adapter file supplied by the user. The adapter must expose
    build_backend(config) or GeoDTRAdapter(config), and the returned object must
    provide encode_many(images_bgr) -> np.ndarray with shape [N, D].
    """

    def __init__(self):
        self.adapter = None
        self.load_model()

    def _adapter_path(self):
        adapter_path = getattr(C, "GEODTR_ADAPTER_PATH", "")
        if adapter_path:
            return os.path.abspath(adapter_path)
        if C.GEODTR_REPO_DIR:
            candidate = os.path.join(C.GEODTR_REPO_DIR, "codex_geodtr_adapter.py")
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        return ""

    def _load_adapter_module(self, adapter_path):
        spec = importlib.util.spec_from_file_location("codex_geodtr_adapter", adapter_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import GeoDTR adapter: {adapter_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load_model(self):
        if not C.GEODTR_REPO_DIR or not os.path.isdir(C.GEODTR_REPO_DIR):
            raise FileNotFoundError(
                "GEODTR_REPO_DIR is empty or does not exist. Put the official GeoDTR+/GeoGTR "
                "repository on disk and set GEODTR_REPO_DIR in config.py."
            )
        if not os.path.exists(C.GEODTR_WEIGHT_DIR):
            raise FileNotFoundError(
                "GEODTR_WEIGHT_DIR does not exist. Download the official pretrained weights "
                "and set GEODTR_WEIGHT_DIR in config.py."
            )

        adapter_path = self._adapter_path()
        if not adapter_path or not os.path.isfile(adapter_path):
            raise FileNotFoundError(
                "GEODTR_ADAPTER_PATH is not configured. Copy "
                "adapters/geodtr_plus_adapter_template.py, fill in the official model loading "
                "code, and set GEODTR_ADAPTER_PATH in config.py."
            )

        if C.GEODTR_REPO_DIR not in sys.path:
            sys.path.insert(0, C.GEODTR_REPO_DIR)

        module = self._load_adapter_module(adapter_path)
        if hasattr(module, "build_backend"):
            self.adapter = module.build_backend(C)
        elif hasattr(module, "GeoDTRAdapter"):
            self.adapter = module.GeoDTRAdapter(C)
        else:
            raise AttributeError(
                "GeoDTR adapter must define build_backend(config) or GeoDTRAdapter(config)."
            )
        if not hasattr(self.adapter, "encode_many"):
            raise AttributeError("GeoDTR adapter object must define encode_many(images_bgr).")

    def encode_many(self, imgs):
        feats = self.adapter.encode_many(imgs)
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[0] != len(imgs):
            raise ValueError(
                f"GeoDTR adapter must return a [N, D] feature matrix. Got {feats.shape}, N={len(imgs)}."
            )
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        return feats

    def encode_query(self, imgs):
        if hasattr(self.adapter, "encode_query"):
            feats = self.adapter.encode_query(imgs)
            feats = np.asarray(feats, dtype=np.float32)
            feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
            return feats
        return self.encode_many(imgs)

    def encode_gallery(self, imgs):
        if hasattr(self.adapter, "encode_gallery"):
            feats = self.adapter.encode_gallery(imgs)
            feats = np.asarray(feats, dtype=np.float32)
            feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
            return feats
        return self.encode_many(imgs)


def get_backend():
    name = C.FEATURE_BACKEND.lower()
    if name == "opencv":
        return OpenCVFeatureBackend()
    if name in ("geodtr", "geodtr_plus", "geogtr"):
        return ExternalGeoDTRFeatureBackend()
    raise ValueError(f"Unknown FEATURE_BACKEND: {C.FEATURE_BACKEND}")


def cosine_scores(query_feat, gallery_feats):
    q = query_feat.reshape(1, -1)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    g = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)
    return (g @ q.T).reshape(-1)
