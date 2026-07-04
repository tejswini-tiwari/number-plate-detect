"""
Image preprocessing utilities.
"""

from typing import Tuple, Optional
import numpy as np


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to target dimensions.

    Args:
        image: Input image.
        target_size: Desired (width, height).

    Returns:
        Resized image.
    """
    pass


def enhance_contrast(image: np.ndarray, method: str = "clahe") -> np.ndarray:
    """
    Enhance image contrast for better plate visibility.

    Args:
        image: Input image.
        method: Enhancement method ("clahe", "histogram", "adaptive").

    Returns:
        Contrast-enhanced image.
    """
    pass


def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    Remove noise from an image.

    Args:
        image: Input image.

    Returns:
        Denoised image.
    """
    pass


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust image brightness.

    Args:
        image: Input image.
        factor: Brightness adjustment factor (>1 brightens, <1 darkens).

    Returns:
        Adjusted image.
    """
    pass


def crop_and_pad(image: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 10) -> np.ndarray:
    """
    Crop a region from image with optional padding.

    Args:
        image: Input image.
        bbox: Bounding box as (x1, y1, x2, y2).
        padding: Extra pixels to add around the crop.

    Returns:
        Cropped image region.
    """
    pass

def shrink_box(x1: int, y1: int, x2: int, y2: int, shrink_pct: float = 0.08) -> tuple:
    """
    Shrink a bounding box inward on all sides by a percentage of its size.

    Useful when the detector's box is slightly loose and includes
    surrounding vehicle body / badges / text alongside the actual plate,
    which then gets picked up by OCR. Shrinking pulls the crop edges in
    toward the plate itself.

    Args:
        x1, y1, x2, y2: Original bounding box coordinates.
        shrink_pct: Fraction of width/height to shrink from each side
            (0.08 = 8% off each side, ~16% total per dimension).

    Returns:
        (x1, y1, x2, y2) shrunk box, clamped to remain valid.
    """
    width = x2 - x1
    height = y2 - y1

    dx = int(width * shrink_pct)
    dy = int(height * shrink_pct)

    new_x1 = x1 + dx
    new_y1 = y1 + dy
    new_x2 = x2 - dx
    new_y2 = y2 - dy

    # Guard against over-shrinking tiny boxes into invalid/zero-size crops
    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return x1, y1, x2, y2

    return new_x1, new_y1, new_x2, new_y2

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale.

    Args:
        image: Input image (BGR or RGB).

    Returns:
        Grayscale image.
    """
    pass


def apply_threshold(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Apply thresholding to binarize image.

    Args:
        image: Input grayscale image.
        method: Thresholding method ("otsu", "adaptive", "manual").

    Returns:
        Binary image.
    """
    pass