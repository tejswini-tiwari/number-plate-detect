"""
YOLO-based number plate detection.
"""

import logging
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
import cv2

from ultralytics import YOLO

# Get logger for this module
logger = logging.getLogger(__name__)


class PlateDetector:
    """Detects vehicle number plates in images using YOLO."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the detector with a YOLO model.

        Args:
            model_path: Path to the YOLO model weights file.
                      Defaults to pretrained YOLOv8n if not provided.
        """
        if model_path is None:
            model_path = "yolov8n.pt"

        logger.info(f"Initializing YOLO model from path: {model_path}")
        try:
            self._model = YOLO(model_path)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {model_path}: {e}")
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")

    def detect(self, image: Union[np.ndarray, str]) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect number plates in an image.

        Args:
            image: Input image as numpy array or path to image file.

        Returns:
            List of bounding boxes as (x1, y1, x2, y2, confidence) tuples.
        """
        if isinstance(image, str):
            logger.debug(f"Reading image from path: {image}")
            img_arr = cv2.imread(image)
            if img_arr is None:
                logger.error(f"Cannot read image file: {image}")
                raise ValueError(f"Cannot read image file: {image}")
            image = img_arr
        elif not isinstance(image, np.ndarray):
            logger.error("Invalid image type provided. Must be numpy array or file path.")
            raise TypeError("image must be numpy array or file path")

        results = self._model(image)
        boxes = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, conf))

        if not boxes:
            logger.info("No number plates detected in the image.")
        else:
            logger.info(f"Detected {len(boxes)} number plate(s).")

        return boxes

    def detect_with_classes(self, image: Union[np.ndarray, str]) -> List[Dict]:
        """
        Detect number plates and return with class information.

        Args:
            image: Input image as numpy array or path to image file.

        Returns:
            List of detection dictionaries with bbox, confidence, and class_id.
        """
        if isinstance(image, str):
            logger.debug(f"Reading image from path: {image}")
            img_arr = cv2.imread(image)
            if img_arr is None:
                logger.error(f"Cannot read image file: {image}")
                raise ValueError(f"Cannot read image file: {image}")
            image = img_arr
        elif not isinstance(image, np.ndarray):
            logger.error("Invalid image type provided. Must be numpy array or file path.")
            raise TypeError("image must be numpy array or file path")

        results = self._model(image)
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class_id": cls_id,
                })

        if not detections:
            logger.info("No number plates detected in the image.")
        else:
            logger.info(f"Detected {len(detections)} number plate(s) with classes.")

        return detections

    def draw_boxes(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int, float]],
        thickness: int = 2,
        text_color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draw bounding boxes on the image for visualization.

        Args:
            image: Input image as numpy array.
            boxes: List of bounding boxes as (x1, y1, x2, y2, confidence).
            thickness: Line thickness for boxes.
            text_color: RGB color for the label text.

        Returns:
            Image with drawn boxes.
        """
        output = image.copy()

        for box in boxes:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(output, (x1, y1), (x2, y2), text_color, thickness)
            label = f"{conf:.2f}"
            cv2.putText(
                output,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                thickness,
            )

        return output

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded YOLO model.

        Returns:
            Dictionary with model metadata.
        """
        return {
            "model_name": self._model.model_name if hasattr(self._model, "model_name") else "unknown",
            "task": getattr(self._model, "task", "detect"),
        }