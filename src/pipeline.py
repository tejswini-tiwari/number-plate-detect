"""
End-to-end detection + OCR pipeline.
"""

import os
import csv
import cv2
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np

from src.detector import PlateDetector
from src.ocr_reader import PlateOCR


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
            with open(self._results_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "image_name",
                    "plate_text",
                    "bbox",
                    "detection_confidence",
                    "ocr_confidence",
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
            List of dicts with keys: bbox, plate_text, detection_confidence, ocr_confidence.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        detections = self._detector.detect(image)
        results = []

        crops_dir = self._output_dir / "crops" if save_crops else None
        if crops_dir:
            crops_dir.mkdir(parents=True, exist_ok=True)

        for idx, (x1, y1, x2, y2, det_conf) in enumerate(detections):
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            plate_text, ocr_conf = self._ocr.read_plate(crop)

            result = {
                "bbox": (x1, y1, x2, y2),
                "plate_text": plate_text,
                "detection_confidence": det_conf,
                "ocr_confidence": ocr_conf,
            }
            results.append(result)

            if crops_dir:
                crop_path = crops_dir / f"{Path(image_path).stem}_plate_{idx}.jpg"
                cv2.imwrite(str(crop_path), crop)

        if save_annotated and results:
            annotated = self._draw_annotations(image.copy(), results)
            output_path = self._output_dir / f"{Path(image_path).stem}_annotated.jpg"
            cv2.imwrite(str(output_path), annotated)

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

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{plate_text} | D:{det_conf:.2f} O:{ocr_conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1 - 10, label_size[1])

            cv2.rectangle(
                image,
                (x1, label_y - label_size[1] - 4),
                (x1 + label_size[0], label_y + 4),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                image,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
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
            for r in results:
                writer.writerow([
                    timestamp,
                    image_name,
                    r["plate_text"],
                    r["bbox"],
                    f"{r['detection_confidence']:.4f}",
                    f"{r['ocr_confidence']:.4f}",
                ])

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

            plate_text, ocr_conf = self._ocr.read_plate(crop)

            results.append({
                "bbox": (x1, y1, x2, y2),
                "plate_text": plate_text,
                "detection_confidence": det_conf,
                "ocr_confidence": ocr_conf,
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
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = skip_frames

        frame_idx = 0
        last_processed_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
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

            cv2.imshow(window_name, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()

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
            cv2.putText(
                panel,
                f"{i + 1}. {r['plate_text']}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
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

        for r in results:
            plate_text = r["plate_text"]
            if not plate_text:
                continue

            key = (plate_text,)
            logged_time = None

            for logged_plate, ts in self._logged_plates:
                if logged_plate == plate_text:
                    logged_time = ts
                    break

            if logged_time and (now - logged_time) < dedup_window:
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
                ])

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Process a video file and save annotated output.

        Args:
            video_path: Path to input video.
            output_path: Path for output video (optional).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_path is None:
            output_path = str(self._output_dir / f"{Path(video_path).stem}_result.mp4")

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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Vehicle Number Plate Detection and Recognition"
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--image", type=str, help="Path to input image file")
    mode_group.add_argument("--video", type=str, help="Path to input video file")
    mode_group.add_argument("--webcam", action="store_true", help="Use webcam as video source")
    parser.add_argument("--webcam-index", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLO model weights")
    parser.add_argument("--ocr-gpu", action="store_true", help="Enable GPU for OCR")
    parser.add_argument("--skip-frames", type=int, default=5, help="Process every N frames (default: 5)")
    parser.add_argument("--dedup-seconds", type=float, default=3.0, help="Deduplication window in seconds")
    parser.add_argument("--no-display", action="store_true", help="Disable live display")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    pipeline = PlateRecognitionPipeline(
        model_path=args.model,
        ocr_gpu=args.ocr_gpu,
        output_dir=args.output_dir,
    )

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {args.image}")
            return
        results = pipeline.process_image(str(image_path))
        print(f"\nResults for: {args.image}")
        print("-" * 60)
        if not results:
            print("No plates detected.")
        for i, r in enumerate(results, 1):
            print(f"Plate {i}: {r['plate_text']} (D:{r['detection_confidence']:.2f} O:{r['ocr_confidence']:.2f})")

    elif args.webcam:
        pipeline.process_video_live(
            video_source=args.webcam_index,
            skip_frames=args.skip_frames,
            dedup_seconds=args.dedup_seconds,
            window_name="Webcam - Press Q to quit" if not args.no_display else "",
        )

    elif args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video not found: {args.video}")
            return
        pipeline.process_video_live(
            video_source=str(video_path),
            skip_frames=args.skip_frames,
            dedup_seconds=args.dedup_seconds,
            window_name=f"Video: {video_path.name} - Press Q to quit" if not args.no_display else "",
        )


if __name__ == "__main__":
    main()


NumberPlatePipeline = PlateRecognitionPipeline