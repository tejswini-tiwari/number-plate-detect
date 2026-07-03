"""
Number Plate Detection and Recognition Pipeline - CLI Entrypoint
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Setup logging system
def setup_logging(output_dir: str = "outputs") -> None:
    """Configures the logging system to output to both console and log file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = Path(output_dir) / "app.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

from src.pipeline import PlateRecognitionPipeline

logger = logging.getLogger("main")


def print_results(results, image_path):
    """Print detection results to console."""
    logger.info(f"Summary of results for: {image_path}")
    print(f"\nResults for: {image_path}")
    print("-" * 60)

    if not results:
        print("No plates detected.")
        logger.info("No plates detected in print_results summary.")
        return

    for i, r in enumerate(results, 1):
        low_conf_str = " [LOW OCR CONFIDENCE]" if r.get('low_confidence') else ""
        print(f"Plate {i}{low_conf_str}:")
        print(f"  Text:     {r['plate_text']}")
        print(f"  BBox:     {r['bbox']}")
        print(f"  Det Conf: {r['detection_confidence']:.4f}")
        print(f"  OCR Conf: {r['ocr_confidence']:.4f}")
        logger.info(f"Plate {i} - Text: {r['plate_text']}, Det Conf: {r['detection_confidence']:.4f}, OCR Conf: {r['ocr_confidence']:.4f}")


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

    # Initialize logging
    setup_logging(args.output_dir)
    logger.info("Starting Number Plate Recognition CLI...")

    try:
        pipeline = PlateRecognitionPipeline(
            model_path=args.model,
            ocr_gpu=args.ocr_gpu,
            output_dir=args.output_dir,
        )
    except Exception as e:
        logger.critical(f"Failed to initialize pipeline: {e}", exc_info=True)
        sys.exit(1)

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"Image not found: {args.image}")
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        
        try:
            results = pipeline.process_image(str(image_path))
            print_results(results, args.image)
        except Exception as e:
            logger.error(f"Error processing image {args.image}: {e}", exc_info=True)
            sys.exit(1)

    elif args.webcam:
        logger.info("Starting webcam-based detection...")
        try:
            pipeline.process_video_live(
                video_source=args.webcam_index,
                skip_frames=args.skip_frames,
                dedup_seconds=args.dedup_seconds,
                window_name="Webcam - Press Q to quit",
            )
        except Exception as e:
            logger.error(f"Error during webcam live processing: {e}", exc_info=True)
            sys.exit(1)

    elif args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Video file not found: {args.video}")
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)
        
        logger.info(f"Starting video processing for: {args.video}")
        try:
            pipeline.process_video_live(
                video_source=str(video_path),
                skip_frames=args.skip_frames,
                dedup_seconds=args.dedup_seconds,
                window_name=f"Video: {video_path.name} - Press Q to quit",
            )
        except Exception as e:
            logger.error(f"Error during video processing: {e}", exc_info=True)
            sys.exit(1)

    logger.info("Execution complete.")


if __name__ == "__main__":
    main()