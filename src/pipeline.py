"""
End-to-end detection + OCR pipeline.
"""

import os
import csv
import cv2
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np

from src.detector import PlateDetector
from src.ocr_reader import PlateOCR

# Get logger for this module
logger = logging.getLogger(__name__)


class PlateRecognitionPipeline:
    """Combines YOLO detection with OCR for end-to-end plate recognition."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        ocr_languages: List[str] = ["en"],
        ocr_gpu: bool = False,
        output_dir: str = "outputs",
    ):
        """
        Initialize the complete pipeline.

        Args:
            model_path: Path to YOLO model weights (defaults to yolov8n.pt).
            ocr_languages: Language codes for OCR.
            ocr_gpu: Whether to use GPU for OCR.
            output_dir: Directory for saving outputs.
        """
        logger.info("Initializing PlateRecognitionPipeline...")
        self._detector = PlateDetector(model_path)
        self._ocr = PlateOCR(languages=ocr_languages, gpu=ocr_gpu)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._results_csv = self._output_dir / "results.csv"
        self._init_results_csv()

        self._last_detections: List[Dict] = []
        self._logged_plates: List[tuple] = []
        self._dedup_seconds: float = 3.0

    def _init_results_csv(self) -> None:
        """Initialize results CSV with headers if it doesn't exist."""
        if not self._results_csv.exists():
            logger.info(f"Creating results CSV file at {self._results_csv}")
            with open(self._results_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "image_name",
                    "plate_text",
                    "bbox",
                    "detection_confidence",
                    "ocr_confidence",
                    "low_confidence",
                ])

    def process_image(
        self,
        image_path: str,
        save_annotated: bool = True,
        save_crops: bool = False,
    ) -> List[Dict]:
        """
        Process a single image and return all detected plates with text.

        Args:
            image_path: Path to the input image.
            save_annotated: Whether to save annotated image to outputs/.
            save_crops: Whether to save cropped plate images.

        Returns:
            List of dicts with keys: bbox, plate_text, detection_confidence, ocr_confidence, low_confidence.
        """
        logger.info(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Cannot read image: {image_path}")
            raise ValueError(f"Cannot read image: {image_path}")

        detections = self._detector.detect(image)
        results = []

        if not detections:
            logger.info(f"No license plates detected in image: {image_path}")

        crops_dir = self._output_dir / "crops" if save_crops else None
        if crops_dir:
            crops_dir.mkdir(parents=True, exist_ok=True)

        for idx, (x1, y1, x2, y2, det_conf) in enumerate(detections):
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                logger.warning(f"Empty crop bounding box detected: {(x1, y1, x2, y2)}")
                continue

            plate_text, ocr_conf, is_low_conf = self._ocr.read_plate(crop)

            result = {
                "bbox": (x1, y1, x2, y2),
                "plate_text": plate_text if plate_text else "[NO TEXT]",
                "detection_confidence": det_conf,
                "ocr_confidence": ocr_conf,
                "low_confidence": is_low_conf,
            }
            results.append(result)

            if crops_dir:
                crop_path = crops_dir / f"{Path(image_path).stem}_plate_{idx}.jpg"
                cv2.imwrite(str(crop_path), crop)
                logger.debug(f"Saved cropped plate to {crop_path}")

        if save_annotated and results:
            annotated = self._draw_annotations(image.copy(), results)
            output_path = self._output_dir / f"{Path(image_path).stem}_annotated.jpg"
            cv2.imwrite(str(output_path), annotated)
            logger.info(f"Saved annotated image to {output_path}")

        self._log_results(image_path, results)

        return results

    def _draw_annotations(self, image: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and text labels on the image.

        Args:
            image: Input image.
            results: List of detection results.

        Returns:
            Annotated image.
        """
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            plate_text = r["plate_text"]
            det_conf = r["detection_confidence"]
            ocr_conf = r["ocr_confidence"]
            is_low_conf = r["low_confidence"]

            color = (0, 0, 255) if is_low_conf else (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{plate_text} {'[LOW_CONF]' if is_low_conf else ''} | D:{det_conf:.2f} O:{ocr_conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1 - 10, label_size[1])

            cv2.rectangle(
                image,
                (x1, label_y - label_size[1] - 4),
                (x1 + label_size[0], label_y + 4),
                color,
                -1,
            )
            cv2.putText(
                image,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255) if is_low_conf else (0, 0, 0),
                2,
            )

        return image

    def _log_results(self, image_path: str, results: List[Dict]) -> None:
        """
        Log detection results to CSV file.

        Args:
            image_path: Path to the processed image.
            results: List of detection results.
        """
        timestamp = datetime.now().isoformat()
        image_name = Path(image_path).name

        with open(self._results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not results:
                # Log a case where no plates were detected
                writer.writerow([
                    timestamp,
                    image_name,
                    "[NO PLATES DETECTED]",
                    "()",
                    "0.0000",
                    "0.0000",
                    "True",
                ])
                logger.info(f"Logged empty detection event for {image_name} to CSV.")
            else:
                for r in results:
                    writer.writerow([
                        timestamp,
                        image_name,
                        r["plate_text"],
                        r["bbox"],
                        f"{r['detection_confidence']:.4f}",
                        f"{r['ocr_confidence']:.4f}",
                        str(r["low_confidence"]),
                    ])
                    logger.info(f"Logged plate detection '{r['plate_text']}' to CSV.")

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a single video frame.

        Args:
            frame: Video frame as numpy array.

        Returns:
            List of detected plate info for the frame.
        """
        detections = self._detector.detect(frame)
        results = []

        for x1, y1, x2, y2, det_conf in detections:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            plate_text, ocr_conf, is_low_conf = self._ocr.read_plate(crop)

            results.append({
                "bbox": (x1, y1, x2, y2),
                "plate_text": plate_text if plate_text else "[NO TEXT]",
                "detection_confidence": det_conf,
                "ocr_confidence": ocr_conf,
                "low_confidence": is_low_conf,
            })

        return results

    def process_video_live(
        self,
        video_source: Union[str, int] = 0,
        skip_frames: int = 5,
        dedup_seconds: float = 3.0,
        window_name: str = "Plate Recognition",
    ) -> None:
        """
        Process video stream (file or webcam) with optimized frame processing.

        Runs detection + OCR every N frames to reduce compute load.
        Reuses last known boxes/text on skipped frames.
        Deduplicates same plate text within a time window.

        Args:
            video_source: Video file path or webcam index (0).
            skip_frames: Process every N frames (default 5).
            dedup_seconds: Ignore same plate if seen within this duration.
            window_name: Window title for display.
        """
        logger.info(f"Opening video source: {video_source}")
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {video_source}")
            raise ValueError(f"Cannot open video source: {video_source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = skip_frames

        frame_idx = 0
        last_processed_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream or cannot read frame.")
                break

            if frame_idx % frame_interval == 0:
                results = self.process_frame(frame)
                self._last_detections = results
                last_processed_frame = frame.copy()
                self._dedup_seconds = dedup_seconds
                self._log_unique_results(results)
            else:
                results = self._last_detections

            if last_processed_frame is not None:
                annotated = self._draw_annotations(last_processed_frame.copy(), results)
                display = self._compose_display_frame(annotated, results, frame_idx)
            else:
                display = frame

            if window_name:
                cv2.imshow(window_name, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User interrupted live video stream by pressing 'q'.")
                break

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()
        logger.info("Video stream closed and resources released.")

    def _compose_display_frame(
        self,
        annotated: np.ndarray,
        results: List[Dict],
        frame_idx: int,
    ) -> np.ndarray:
        """
        Compose the display frame with annotated image and info overlay.

        Args:
            annotated: Annotated frame with boxes.
            results: Current detection results.
            frame_idx: Current frame index.

        Returns:
            Display frame.
        """
        h, w = annotated.shape[:2]
        panel_width = 300
        panel = np.ones((h, panel_width, 3), dtype=np.uint8) * 40

        y_offset = 30
        cv2.putText(
            panel,
            f"Frame: {frame_idx}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        y_offset += 30

        cv2.putText(
            panel,
            f"Plates detected: {len(results)}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        y_offset += 40

        for i, r in enumerate(results):
            if y_offset > h - 30:
                break
            text = r['plate_text']
            if r['low_confidence']:
                text += " [LOW_CONF]"
            cv2.putText(
                panel,
                f"{i + 1}. {text}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255) if r['low_confidence'] else (0, 255, 0),
                1,
            )
            y_offset += 25

        display = np.hstack([annotated, panel])
        return display

    def _log_unique_results(self, results: List[Dict]) -> None:
        """
        Log results to CSV, skipping duplicates within dedup window.

        Args:
            results: Detection results for current frame.
        """
        now = datetime.now()
        dedup_window = timedelta(seconds=self._dedup_seconds)

        if not results:
            # Optionally log empty state to app log
            logger.debug("No plates detected in current frame, skipping CSV logging.")
            return

        for r in results:
            plate_text = r["plate_text"]
            if not plate_text or plate_text == "[NO TEXT]":
                continue

            logged_time = None

            for logged_plate, ts in self._logged_plates:
                if logged_plate == plate_text:
                    logged_time = ts
                    break

            if logged_time and (now - logged_time) < dedup_window:
                logger.debug(f"Skipping duplicate log for plate '{plate_text}' within dedup window.")
                continue

            self._logged_plates = [
                (p, t) for p, t in self._logged_plates
                if p != plate_text or (now - t) >= dedup_window
            ]
            self._logged_plates.append((plate_text, now))

            with open(self._results_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    now.isoformat(),
                    "video_stream",
                    plate_text,
                    r["bbox"],
                    f"{r['detection_confidence']:.4f}",
                    f"{r['ocr_confidence']:.4f}",
                    str(r["low_confidence"]),
                ])
            logger.info(f"Logged unique plate detection '{plate_text}' from video stream.")

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Process a video file and save annotated output.

        Args:
            video_path: Path to input video.
            output_path: Path for output video (optional).
        """
        logger.info(f"Offline processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            raise ValueError(f"Cannot open video: {video_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_path is None:
            output_path = str(self._output_dir / f"{Path(video_path).stem}_result.mp4")

        logger.info(f"Saving processed video to: {output_path}")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.process_frame(frame)
            annotated = self._draw_annotations(frame, results)
            out.write(annotated)

            frame_idx += 1

        cap.release()
        out.release()
        logger.info("Video processing complete.")

    def get_pipeline_info(self) -> Dict:
        """
        Get information about the pipeline configuration.

        Returns:
            Dictionary with detector and OCR info.
        """
        return {
            "detector": self._detector.get_model_info(),
            "ocr_languages": self._ocr._reader.langs if hasattr(self._ocr, "_reader") else [],
        }


NumberPlatePipeline = PlateRecognitionPipeline