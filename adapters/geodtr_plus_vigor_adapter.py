# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


class GeoDTRAdapter:
    def __init__(self, config):
        self.config = config
        os.environ.setdefault("USER", "codex")
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.repo_dir = Path(config.GEODTR_REPO_DIR)
        self.weight_dir = Path(config.GEODTR_WEIGHT_DIR)
        self.params = self._read_params()
        self.model = self._load_model()
        self.model.eval()

    def _read_params(self):
        files = sorted(self.weight_dir.glob("*_parameter.json"))
        if not files:
            raise FileNotFoundError(f"No *_parameter.json found in {self.weight_dir}")
        with open(files[0], "r", encoding="utf-8") as f:
            return json.load(f)

    def _best_checkpoint(self):
        epoch_dirs = []
        for p in self.weight_dir.iterdir():
            if p.is_dir() and p.name.startswith("epoch_"):
                try:
                    epoch_dirs.append((int(p.name.split("_")[1]), p))
                except Exception:
                    pass
        if not epoch_dirs:
            raise FileNotFoundError(f"No epoch_* checkpoint folder found in {self.weight_dir}")
        _, folder = sorted(epoch_dirs, reverse=True)[0]
        ckpt = folder / f"{folder.name}.pth"
        if not ckpt.exists():
            candidates = list(folder.glob("*.pth"))
            if not candidates:
                raise FileNotFoundError(f"No .pth checkpoint found in {folder}")
            ckpt = candidates[0]
        return ckpt

    def _load_model(self):
        import sys
        import torchvision.models as tv_models

        repo = str(self.repo_dir.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

        # GeoDTR+ loads torchvision ImageNet weights before loading the project checkpoint.
        # The checkpoint contains the trained weights already, so skip the extra download.
        self._disable_torchvision_weight_downloads(tv_models)

        from models.GeoDTR import GeoDTR

        model = GeoDTR(
            descriptors=self.params["descriptors"],
            tr_heads=self.params["TR_heads"],
            tr_layers=self.params["TR_layers"],
            dropout=self.params["dropout"],
            d_hid=self.params["TR_dim"],
            is_polar=False,
            backbone=self.params["backbone"],
            dataset="VIGOR",
            normalize=self.params["normalize"],
            orthogonalize=self.params["orthogonalize"],
            bottleneck=self.params["bottleneck"],
        )
        model = nn.DataParallel(model)
        model.to(self.device)
        ckpt = torch.load(self._best_checkpoint(), map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        return model

    def _disable_torchvision_weight_downloads(self, tv_models):
        for name in ("convnext_tiny", "efficientnet_b3", "resnet34"):
            if not hasattr(tv_models, name):
                continue
            original = getattr(tv_models, name)
            if getattr(original, "_geodtr_no_download", False):
                continue

            def no_download(*args, _original=original, **kwargs):
                kwargs["weights"] = None
                kwargs.pop("pretrained", None)
                return _original(*args, **kwargs)

            no_download._geodtr_no_download = True
            setattr(tv_models, name, no_download)

    def _preprocess_satellite(self, img_bgr):
        return self._preprocess(img_bgr, width=320, height=320)

    def _preprocess_ground(self, img_bgr):
        return self._preprocess(img_bgr, width=640, height=320)

    def _preprocess(self, img_bgr, width, height):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        arr = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    @torch.no_grad()
    def encode_query(self, images_bgr):
        grd = torch.stack([self._preprocess_ground(img) for img in images_bgr], dim=0).to(self.device)
        sat = torch.zeros((grd.shape[0], 3, 320, 320), dtype=grd.dtype, device=self.device)
        _, grd_global, _, _ = self.model(sat, grd, is_cf=False)
        return grd_global.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_gallery(self, images_bgr):
        sat = torch.stack([self._preprocess_satellite(img) for img in images_bgr], dim=0).to(self.device)
        grd = torch.zeros((sat.shape[0], 3, 320, 640), dtype=sat.dtype, device=self.device)
        sat_global, _, _, _ = self.model(sat, grd, is_cf=False)
        return sat_global.detach().cpu().numpy().astype(np.float32)

    def encode_many(self, images_bgr):
        return self.encode_gallery(images_bgr)


def build_backend(config):
    return GeoDTRAdapter(config)
