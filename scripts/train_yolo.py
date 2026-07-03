"""
Fine-tune YOLOv8 on license plate detection dataset.

Usage:
    python scripts/train_yolo.py --data data/plate_dataset/data.yaml --epochs 100
    python scripts/train_yolo.py --data data/plate_dataset/data.yaml -e 50 -b 16 -s 640
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

def get_default_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "0"
    return "cpu"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for license plate detection"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml configuration file",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch",
        "-b",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        "-s",
        type=int,
        default=640,
        help="Image size for training (default: 640)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolov8n.pt",
        help="Pretrained model to start from (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output model path (default: models/plate_yolov8.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        help="Device to use (default: auto-detect, or 'cpu', '0', '1', etc.)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers (default: 8)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_path = args.output or "models/plate_yolov8.pt"
    os.makedirs(os.path.dirname(output_path) or "models", exist_ok=True)

    print(f"Loading pretrained model: {args.model}")
    model = YOLO(args.model)

    print(f"Starting fine-tuning on dataset: {args.data}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch size: {args.batch}")
    print(f"  image size: {args.imgsz}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        resume=args.resume,
        project="outputs",
        name="yolo_training",
        exist_ok=True,
        save=True,
        save_period=10,
    )

    best_model_path = results.save_dir / "weights" / "best.pt"
    if best_model_path.exists():
        import shutil

        shutil.copy(best_model_path, output_path)
        print(f"\nBest model saved to: {output_path}")
    else:
        last_model_path = results.save_dir / "weights" / "last.pt"
        if last_model_path.exists():
            import shutil

            shutil.copy(last_model_path, output_path)
            print(f"\nFinal model saved to: {output_path}")

    print("\nTraining complete!")
    print(f"Training results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()