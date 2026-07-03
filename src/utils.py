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