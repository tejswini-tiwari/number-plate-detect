"""
OCR-based plate number extraction.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import cv2
import re
import easyocr

# Get logger for this module
logger = logging.getLogger(__name__)


class PlateOCR:
    """Extracts text from detected number plates using EasyOCR."""

    def __init__(self, languages: List[str] = ["en"], gpu: bool = False, conf_threshold: float = 0.5):
        """
        Initialize the OCR reader.

        Args:
            languages: List of language codes for OCR (e.g., ["en", "ar"]).
            gpu: Whether to use GPU acceleration for OCR.
            conf_threshold: Threshold below which results are flagged as low confidence.
        """
        logger.info(f"Initializing EasyOCR reader with languages {languages} (GPU={gpu})")
        self._reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
        self._conf_threshold = conf_threshold

    def preprocess(self, cropped_plate_image: np.ndarray) -> np.ndarray:
        """
        Preprocess a cropped plate image to improve OCR accuracy.

        Converts to grayscale, applies adaptive thresholding, and resizes
        the image to enhance text visibility.

        Args:
            cropped_plate_image: Raw cropped plate image (BGR or RGB).

        Returns:
            Preprocessed image ready for OCR.
        """
        logger.debug("Preprocessing cropped plate image for OCR.")
        if len(cropped_plate_image.shape) == 3:
            gray = cv2.cvtColor(cropped_plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_plate_image.copy()

        height, width = gray.shape
        new_width = max(300, int(width * 2))
        new_height = max(100, int(height * 2))
        resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)

        adaptive = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        return adaptive

    def read_plate(self, cropped_plate_image: np.ndarray) -> Tuple[str, float, bool]:
        """
        Preprocess a cropped plate image, run OCR, and return text with confidence and low-confidence flag.

        Args:
            cropped_plate_image: Raw cropped plate image.

        Returns:
            Tuple of (extracted_text, average_confidence, is_low_confidence).
        """
        if cropped_plate_image is None or cropped_plate_image.size == 0:
            logger.warning("Empty cropped plate image passed to read_plate.")
            return "", 0.0, True

        preprocessed = self.preprocess(cropped_plate_image)

        logger.debug("Running EasyOCR on preprocessed image.")
        results = self._reader.readtext(preprocessed)

        if not results:
            logger.info("No text detected on the number plate.")
            return "", 0.0, True

        full_text = ""
        total_conf = 0.0
        count = 0

        for (bbox, text, conf) in results:
            cleaned = self.clean_text(text)
            full_text += cleaned
            total_conf += conf
            count += 1

        avg_conf = total_conf / count if count > 0 else 0.0
        is_low_confidence = avg_conf < self._conf_threshold

        if is_low_confidence:
            logger.warning(f"OCR result '{full_text}' flagged as low confidence ({avg_conf:.2f} < {self._conf_threshold}).")
        else:
            logger.info(f"Successfully recognized plate text: '{full_text}' with confidence: {avg_conf:.2f}")

        return full_text, avg_conf, is_low_confidence

    def clean_text(self, raw_text: str) -> str:
        """
        Clean and format OCR text to match typical license plate patterns.

        Removes non-alphanumeric characters, converts to uppercase,
        and removes spaces to produce a standard plate format.

        Args:
            raw_text: Raw text extracted from OCR.

        Returns:
            Formatted license plate string.
        """
        alphanumeric = re.sub(r"[^a-zA-Z0-9]", "", raw_text)
        cleaned = alphanumeric.upper().strip()
        return cleaned

    def read_text(self, image: np.ndarray) -> str:
        """
        Extract text from a cropped plate image.

        Args:
            image: Cropped plate image as numpy array.

        Returns:
            Extracted text string.
        """
        text, _, _ = self.read_plate(image)
        return text

    def read_text_batch(self, images: List[np.ndarray]) -> List[str]:
        """
        Extract text from multiple plate images.

        Args:
            images: List of cropped plate images.

        Returns:
            List of extracted text strings.
        """
        return [self.read_text(img) for img in images]


OCRReader = PlateOCR