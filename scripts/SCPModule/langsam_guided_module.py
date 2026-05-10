import logging
from typing import Any, Tuple
from pathlib import Path

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESOURCE_ROOT = _REPO_ROOT / "resource"
_DEFAULT_GDINO_SNAPSHOT = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _resize_prob_map(prob_map: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    if prob_map.shape == (th, tw):
        return prob_map.astype(np.float32)
    pil = Image.fromarray((np.clip(prob_map, 0.0, 1.0) * 255.0).astype(np.uint8))
    pil = pil.resize((tw, th), resample=Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


class LangSAMGuidedSegmenter:
    """LangSAM Guided module: output A^- and A^+ strict per-pixel probabilities."""

    def __init__(
        self,
        sam_type: str = "sam2.1_hiera_small",
        sam_ckpt_path: str = str(_RESOURCE_ROOT / "sam2.1_hiera_small.pt"),
        gdino_model_ckpt_path: str = str(
            _RESOURCE_ROOT
            / "models--IDEA-Research--grounding-dino-base"
            / "snapshots"
            / _DEFAULT_GDINO_SNAPSHOT
        ),
        gdino_processor_ckpt_path: str = str(
            _RESOURCE_ROOT
            / "models--IDEA-Research--grounding-dino-base"
            / "snapshots"
            / _DEFAULT_GDINO_SNAPSHOT
        ),
    ):
        from lang_sam import LangSAM

        self.model = LangSAM(
            sam_type=sam_type,
            sam_ckpt_path=sam_ckpt_path,
            gdino_model_ckpt_path=gdino_model_ckpt_path,
            gdino_processor_ckpt_path=gdino_processor_ckpt_path,
        )
        logger.info("LangSAMGuidedSegmenter initialized.")

    def predict_prob_map(self, image_pil: Image.Image, concept: str) -> np.ndarray:
        image_np = np.array(image_pil.convert("RGB"))
        h, w = image_np.shape[:2]
        concept = (concept or "").strip()
        if not concept or concept == "null":
            return np.zeros((h, w), dtype=np.float32)

        try:
            gdino_results = self.model.gdino.predict([image_pil], [concept], box_threshold=0.3, text_threshold=0.25)
            if not gdino_results:
                return np.zeros((h, w), dtype=np.float32)
            boxes = gdino_results[0].get("boxes", None)
            if boxes is None:
                return np.zeros((h, w), dtype=np.float32)
            if hasattr(boxes, "detach"):
                boxes = boxes.detach().cpu().numpy()
            boxes = np.asarray(boxes)
            if boxes.size == 0:
                return np.zeros((h, w), dtype=np.float32)

            _, _, logits_list = self.model.sam.predict_batch([image_np], xyxy=[boxes])
            if not logits_list:
                return np.zeros((h, w), dtype=np.float32)
            logits = np.asarray(logits_list[0])
            if logits.ndim == 2:
                logits = logits[None, ...]
            if logits.ndim != 3:
                return np.zeros((h, w), dtype=np.float32)
            prob = _sigmoid(logits).max(axis=0).astype(np.float32)
            return _resize_prob_map(prob, (h, w))
        except Exception as e:
            logger.warning("LangSAM strict prob failed for concept '%s': %s", concept, e)
            return np.zeros((h, w), dtype=np.float32)

    def predict_before_after(
        self,
        before_image: Image.Image,
        after_image: Image.Image,
        before_concept: str,
        after_concept: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        a_minus = self.predict_prob_map(before_image, before_concept)
        a_plus = self.predict_prob_map(after_image, after_concept)
        return a_minus, a_plus
