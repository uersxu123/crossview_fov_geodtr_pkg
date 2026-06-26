# -*- coding: utf-8 -*-
from functools import lru_cache

import cv2
import numpy as np
import torch

import config as C


def _resize_max_dim(img, max_dim):
    h, w = img.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale >= 1.0:
        return img.copy(), 1.0
    out = cv2.resize(
        img,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return out, scale


def _label_ids(id2label, names):
    wanted = {n.lower() for n in names}
    return [int(i) for i, label in id2label.items() if str(label).lower() in wanted]


class SegFormerPhoneSegmenter:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        self.model_name = getattr(
            C,
            "PHONE_SEMSEG_MODEL",
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        )
        self.max_dim = int(getattr(C, "PHONE_SEMSEG_MAX_DIM", 1536))
        self.device = torch.device(C.DEVICE if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.road_ids = _label_ids(
            self.id2label,
            getattr(C, "PHONE_SEMSEG_ROAD_LABELS", ("road",)),
        )
        self.green_ids = _label_ids(
            self.id2label,
            getattr(C, "PHONE_SEMSEG_GREEN_LABELS", ("vegetation", "terrain")),
        )
        self.ignore_ids = _label_ids(
            self.id2label,
            getattr(
                C,
                "PHONE_SEMSEG_IGNORE_LABELS",
                (
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
                ),
            ),
        )

    @torch.no_grad()
    def segment(self, phone_img):
        h, w = phone_img.shape[:2]
        small, _ = _resize_max_dim(phone_img, self.max_dim)
        sh, sw = small.shape[:2]
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=rgb, return_tensors="pt", do_resize=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(sh, sw),
            mode="bilinear",
            align_corners=False,
        )
        pred_small = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)

        road = np.isin(pred, self.road_ids).astype(np.uint8) * 255
        green = np.isin(pred, self.green_ids).astype(np.uint8) * 255
        ignore = np.isin(pred, self.ignore_ids).astype(np.uint8) * 255

        return {
            "labels": pred,
            "road_mask": road,
            "green_mask": green,
            "ignore_mask": ignore,
            "id2label": self.id2label,
        }


@lru_cache(maxsize=1)
def get_phone_segmenter():
    return SegFormerPhoneSegmenter()


def try_segment_phone(phone_img):
    if getattr(C, "PHONE_SEMSEG_ENABLED", True) is False:
        return None
    try:
        return get_phone_segmenter().segment(phone_img)
    except Exception as exc:
        if getattr(C, "PHONE_SEMSEG_STRICT", False):
            raise
        print(f"[semantic] disabled for this run: {exc}")
        return None
