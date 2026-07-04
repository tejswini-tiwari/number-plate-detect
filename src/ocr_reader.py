"""
OCR-based plate number extraction.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
import re
import easyocr


PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class PlateOCR:
    """Extracts text from detected number plates using EasyOCR."""

    def __init__(self, languages: List[str] = ["en"], gpu: bool = False):
        """
        Initialize the OCR reader.

        Args:
            languages: List of language codes for OCR (e.g., ["en", "ar"]).
            gpu: Whether to use GPU acceleration for OCR.
        """
        self._reader = easyocr.Reader(languages, gpu=gpu, verbose=False)

    def preprocess(self, cropped_plate_image: np.ndarray) -> np.ndarray:
        """
        Preprocess a cropped plate image to improve OCR accuracy.

        Converts to grayscale, denoises, enhances contrast, and resizes.
        Deliberately avoids hard binarization (adaptive/Otsu threshold),
        since deep-learning OCR engines like EasyOCR generally perform
        worse on binarized images than on clean grayscale ones -
        binarization tends to fragment or merge character strokes,
        especially on glossy/reflective plates.

        Args:
            cropped_plate_image: Raw cropped plate image (BGR or RGB).

        Returns:
            Preprocessed grayscale image ready for OCR.
        """
        if len(cropped_plate_image.shape) == 3:
            gray = cv2.cvtColor(cropped_plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_plate_image.copy()

        height, width = gray.shape

        # Upscale moderately - large upscales (2x+) combined with small
        # block-size operations amplify noise rather than helping.
        scale = 2 if max(height, width) < 200 else 1.5
        new_width = max(300, int(width * scale))
        new_height = max(100, int(height * scale))
        resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Mild denoising before contrast enhancement to avoid amplifying
        # sensor/compression noise along with the text.
        denoised = cv2.fastNlMeansDenoising(resized, h=10)

        # CLAHE improves local contrast without collapsing the image to
        # pure black/white, preserving stroke detail for OCR.
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        return enhanced

    def read_plate(
        self,
        cropped_plate_image: np.ndarray,
        allowlist: Optional[str] = PLATE_ALLOWLIST,
    ) -> Tuple[str, float]:
        """
        Preprocess a cropped plate image, run OCR, and return text with confidence.

        Args:
            cropped_plate_image: Raw cropped plate image.
            allowlist: Characters OCR is allowed to output. Restricting this
                to plate-valid characters (A-Z, 0-9) reduces misreads caused
                by visually similar symbols/punctuation. Pass None to disable.

        Returns:
            Tuple of (extracted_text, average_confidence).
        """
        preprocessed = self.preprocess(cropped_plate_image)

        read_kwargs = {"detail": 1, "paragraph": False}
        if allowlist:
            read_kwargs["allowlist"] = allowlist

        results = self._reader.readtext(preprocessed, **read_kwargs)

        if not results:
            return "", 0.0

        full_text = ""
        total_conf = 0.0
        count = 0

        for (bbox, text, conf) in results:
            cleaned = self.clean_text(text)
            if not cleaned:
                continue
            full_text += cleaned
            total_conf += conf
            count += 1

        avg_conf = total_conf / count if count > 0 else 0.0

        return full_text, avg_conf

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
        text, _ = self.read_plate(image)
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