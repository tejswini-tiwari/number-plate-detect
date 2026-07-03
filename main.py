"""
Number Plate Detection and Recognition Pipeline
"""

import argparse
import sys
from pathlib import Path

from src.pipeline import PlateRecognitionPipeline


def print_results(results, image_path):
    """Print detection results to console."""
    print(f"\nResults for: {image_path}")
    print("-" * 60)

    if not results:
        print("No plates detected.")
        return

    for i, r in enumerate(results, 1):
        print(f"Plate {i}:")
        print(f"  Text:     {r['plate_text']}")
        print(f"  BBox:     {r['bbox']}")
        print(f"  Det Conf: {r['detection_confidence']:.4f}")
        print(f"  OCR Conf: {r['ocr_confidence']:.4f}")


def main():
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
            sys.exit(1)
        results = pipeline.process_image(str(image_path))
        print_results(results, args.image)

    elif args.webcam:
        pipeline.process_video_live(
            video_source=args.webcam_index,
            skip_frames=args.skip_frames,
            dedup_seconds=args.dedup_seconds,
            window_name="Webcam - Press Q to quit",
        )

    elif args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)
        pipeline.process_video_live(
            video_source=str(video_path),
            skip_frames=args.skip_frames,
            dedup_seconds=args.dedup_seconds,
            window_name=f"Video: {video_path.name} - Press Q to quit",
        )


if __name__ == "__main__":
    main()