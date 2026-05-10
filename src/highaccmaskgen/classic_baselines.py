import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def absdiff_mask(real: np.ndarray, forged: np.ndarray, threshold: int = 25) -> np.ndarray:
    diff = cv2.absdiff(real, forged)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    return (gray >= threshold).astype(np.uint8)


def ssim_diff_mask(real: np.ndarray, forged: np.ndarray, threshold: int = 30) -> np.ndarray:
    real_g = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY)
    forged_g = cv2.cvtColor(forged, cv2.COLOR_RGB2GRAY)
    _, diff = ssim(real_g, forged_g, full=True)
    diff = ((1.0 - diff) * 255).astype(np.uint8)
    return (diff >= threshold).astype(np.uint8)


def otsu_mask(real: np.ndarray, forged: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(real, forged)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (bw > 0).astype(np.uint8)


def adaptive_mask(real: np.ndarray, forged: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(real, forged)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5
    )
    return (bw > 0).astype(np.uint8)


def morphology_refine(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    opened = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return (closed > 0).astype(np.uint8)
