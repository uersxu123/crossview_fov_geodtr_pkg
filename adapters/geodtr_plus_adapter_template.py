# -*- coding: utf-8 -*-
"""Template adapter for plugging an official GeoDTR+/GeoGTR repo into this project.

Copy this file, fill in the TODO blocks using the official repository's
test.py or test_vigor.py, then set GEODTR_ADAPTER_PATH in config.py.

Required contract:
    build_backend(config) -> object with encode_many(images_bgr)
    encode_many(images_bgr) -> np.ndarray of shape [N, D]
"""
from pathlib import Path

import cv2
import numpy as np
import torch


class GeoDTRAdapter:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.image_size = getattr(config, "GEODTR_INPUT_SIZE", 384)
        self.model = self._load_model().to(self.device)
        self.model.eval()

    def _load_model(self):
        repo_dir = Path(self.config.GEODTR_REPO_DIR)
        weight_dir = Path(self.config.GEODTR_WEIGHT_DIR)
        if not repo_dir.exists():
            raise FileNotFoundError(f"GEODTR_REPO_DIR does not exist: {repo_dir}")
        if not weight_dir.exists():
            raise FileNotFoundError(f"GEODTR_WEIGHT_DIR does not exist: {weight_dir}")

        # TODO:
        # 1. Import the official model class from GEODTR_REPO_DIR.
        # 2. Build the same model used by the official test.py/test_vigor.py.
        # 3. Load the checkpoint from weight_dir.
        #
        # Example shape only:
        #   from model import GeoDTR
        #   model = GeoDTR(...)
        #   ckpt = torch.load(weight_dir / "model.pth", map_location=self.device)
        #   model.load_state_dict(ckpt["state_dict"], strict=False)
        #   return model
        raise NotImplementedError(
            "Fill adapters/geodtr_plus_adapter_template.py with the official "
            "GeoDTR+/GeoGTR model construction and checkpoint loading code."
        )

    def _preprocess(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        arr = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    @torch.no_grad()
    def encode_many(self, images_bgr):
        batch = torch.stack([self._preprocess(img) for img in images_bgr], dim=0).to(self.device)

        # TODO:
        # Replace this with the official feature extraction call.
        # Typical possibilities in cross-view codebases:
        #   feat = self.model(batch)
        #   feat = self.model.get_feature(batch)
        #   feat = self.model.query_net(batch)
        #   feat = self.model.satellite_net(batch)
        #
        # This project currently calls the same encode_many() for phone query
        # and satellite patches. If the official repo has separate branches,
        # adapt this wrapper to expose the right branch for each image type.
        feat = self.model(batch)

        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feat = torch.nn.functional.normalize(feat.float(), dim=1)
        return feat.detach().cpu().numpy().astype(np.float32)

    def encode_query(self, images_bgr):
        # Optional: use the ground/query branch here if the official model has one.
        return self.encode_many(images_bgr)

    def encode_gallery(self, images_bgr):
        # Optional: use the satellite/gallery branch here if the official model has one.
        return self.encode_many(images_bgr)


def build_backend(config):
    return GeoDTRAdapter(config)
