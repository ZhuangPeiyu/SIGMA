from __future__ import annotations

import io
import random

import cv2
import numpy as np


def _random_affine(img: np.ndarray, max_translate: int, max_rotate_deg: float, max_scale_delta: float) -> np.ndarray:
    h, w = img.shape[:2]
    tx = random.uniform(-max_translate, max_translate)
    ty = random.uniform(-max_translate, max_translate)
    angle = random.uniform(-max_rotate_deg, max_rotate_deg)
    scale = 1.0 + random.uniform(-max_scale_delta, max_scale_delta)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += [tx, ty]
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def _jpeg_recompress(img: np.ndarray, qmin: int, qmax: int) -> np.ndarray:
    q = random.randint(qmin, qmax)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def pixel_drift_augment(
    real: np.ndarray,
    forged: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    real_aug = _random_affine(real, cfg["max_translate"], cfg["max_rotate_deg"], cfg["max_scale_delta"])
    forged_aug = _random_affine(forged, cfg["max_translate"], cfg["max_rotate_deg"], cfg["max_scale_delta"])
    if random.random() < cfg["blur_prob"]:
        k = random.choice([3, 5])
        forged_aug = cv2.GaussianBlur(forged_aug, (k, k), 0)
    forged_aug = _jpeg_recompress(forged_aug, cfg["jpeg_quality_min"], cfg["jpeg_quality_max"])
    return real_aug, forged_aug
