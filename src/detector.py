"""
YOLO-based number plate detection.
"""

from typing import List, Tuple, Dict, Union, Optional
import numpy as np
import cv2

from ultralytics import YOLO


class PlateDetector:
    """Detects vehicle number plates in images using YOLO."""

    def __init__(self, model_path: Optional[str] = None, default_conf: float = 0.5):
        """
        Initialize the detector with a YOLO model.

        Args:
            model_path: Path to the YOLO model weights file.
                      Defaults to pretrained YOLOv8n if not provided.
            default_conf: Default confidence threshold used when `detect()`
                      is called without an explicit `conf` argument.
        """
        if model_path is None:
            model_path = "yolov8n.pt"

        self._default_conf = default_conf

        try:
            self._model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")

    @staticmethod
    def _load_image(image: Union[np.ndarray, str]) -> np.ndarray:
        """Load an image from a numpy array or file path, raising on failure."""
        if isinstance(image, str):
            loaded = cv2.imread(image)
            if loaded is None:
                raise ValueError(f"Cannot read image file: {image}")
            return loaded
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise TypeError("image must be numpy array or file path")

    def detect(
        self,
        image: Union[np.ndarray, str],
        conf: Optional[float] = None,
        iou: float = 0.45,
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect number plates in an image.

        Args:
            image: Input image as numpy array or path to image file.
            conf: Confidence threshold (0-1). Detections below this are discarded.
                  Defaults to the detector's `default_conf` if not provided.
            iou: IoU threshold used for non-max suppression.

        Returns:
            List of bounding boxes as (x1, y1, x2, y2, confidence) tuples.
        """
        image = self._load_image(image)
        conf_threshold = conf if conf is not None else self._default_conf

        results = self._model(image, conf=conf_threshold, iou=iou, verbose=False)
        boxes = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_conf = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, box_conf))

        return boxes

    def detect_with_classes(
        self,
        image: Union[np.ndarray, str],
        conf: Optional[float] = None,
        iou: float = 0.45,
    ) -> List[Dict]:
        """
        Detect number plates and return with class information.

        Args:
            image: Input image as numpy array or path to image file.
            conf: Confidence threshold (0-1). Defaults to `default_conf`.
            iou: IoU threshold used for non-max suppression.

        Returns:
            List of detection dictionaries with bbox, confidence, and class_id.
        """
        image = self._load_image(image)
        conf_threshold = conf if conf is not None else self._default_conf

        results = self._model(image, conf=conf_threshold, iou=iou, verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": box_conf,
                    "class_id": cls_id,
                })

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
            "default_conf": self._default_conf,
        }