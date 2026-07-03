"""
Dataset preparation script for license plate detection.

This script generates a data.yaml configuration file required by Ultralytics YOLO
for training on a license plate detection dataset in YOLO format.

Dataset expected structure:
    data/plate_dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Where label files contain YOLO-format bounding boxes:
    class_id x_center y_center width height
    (normalized to 0-1)

================================================================================
OPEN SOURCE LICENSE PLATE DATASETS
================================================================================

1. Roboflow Universe (recommended)
   - Search for "license plate" or "vehicle registration" at:
     https://universe.roboflow.com/
   - Export in "YOLO v5/v8 PyTorch" format
   - Many datasets are CC BY 4.0 licensed

2. AI Hub - Vehicle Registration Plate Dataset
   - https://aihub.or.kr/activity/data
   - May need format conversion

3. Kaggle - License Plate Detection Dataset
   - https://www.kaggle.com/datasets/andrewmvd/license-plate-detection
   - Note: This dataset is in VOC format (XML annotations), not YOLO.
     You would need to convert it using tools like:
     - https://github.com/karatsuba/roboflow-to-yolo
     - LabelImg's built-in format conversion

4. OpenALPR datasets
   - http://www.openalpr.com/datasets.html
   - US and EU plate datasets available

================================================================================
CONVERTING OTHER FORMATS TO YOLO
================================================================================

If your dataset is in a different format (VOC XML, COCO JSON, etc.), use one of:

1. Roboflow's online converter (easiest):
   - Upload to Roboflow, then export in YOLO format

2. Python script with labelImg or similar:
   - labelImg supports switching between formats

3. Custom conversion script example for VOC XML:
   ```python
   import xml.etree.ElementTree as ET
   import os

   def voc_to_yolo(xml_path, img_width, img_height):
       tree = ET.parse(xml_path)
       root = tree.getroot()
       for obj in root.findall("object"):
           bbox = obj.find("bndbox")
           x1 = int(bbox.find("xmin").text)
           y1 = int(bbox.find("ymin").text)
           x2 = int(bbox.find("xmax").text)
           y2 = int(bbox.find("ymax").text)
           x_center = ((x1 + x2) / 2) / img_width
           y_center = ((y1 + y2) / 2) / img_height
           width = (x2 - x1) / img_width
           height = (y2 - y1) / img_height
           # Write: class_id x_center y_center width height
   ```
"""

import os
import yaml
from pathlib import Path
from typing import Optional


def generate_data_yaml(
    dataset_root: str = "data/plate_dataset",
    output_path: str = "data/plate_dataset/data.yaml",
    class_name: str = "license_plate",
    nc: int = 1,
) -> str:
    """
    Generate a data.yaml configuration file for YOLO training.

    Args:
        dataset_root: Root directory of the YOLO format dataset.
        output_path: Where to save the generated data.yaml file.
        class_name: Name of the detection class.
        nc: Number of classes (default 1 for license_plate).

    Returns:
        Path to the generated data.yaml file.
    """
    if not Path(dataset_root).exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}\n"
            "Please create the dataset directory with train/val/test splits first."
        )

    data_config = {
        "path": str(Path(dataset_root).resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": nc,
        "names": {0: class_name},
    }

    with open(output_path, "w") as f:
        yaml.dump(data_config, f, sort_keys=False)

    print(f"Generated data.yaml at: {output_path}")
    print(f"\nContent:")
    print(yaml.dump(data_config, sort_keys=False))

    return output_path


def verify_dataset_structure(dataset_root: str) -> bool:
    """
    Verify that the dataset directory structure is correct.

    Args:
        dataset_root: Root directory of the dataset.

    Returns:
        True if structure is valid, False otherwise.
    """
    required_dirs = [
        "train/images",
        "train/labels",
        "val/images",
        "val/labels",
        "test/images",
        "test/labels",
    ]

    missing = []
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_root, dir_path)
        if not os.path.exists(full_path):
            missing.append(dir_path)

    if missing:
        print("Missing required directories:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease create the dataset structure as follows:")
        print(f"  {dataset_root}/")
        for d in required_dirs:
            print(f"    ├── {d}/")
        return False

    print(f"Dataset structure verified at: {dataset_root}")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate data.yaml for YOLO training"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/plate_dataset",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for data.yaml (default: dataset_root/data.yaml)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset structure before generating yaml",
    )

    args = parser.parse_args()

    if args.verify:
        if not verify_dataset_structure(args.dataset_root):
            return

    output_path = args.output or os.path.join(args.dataset_root, "data.yaml")

    try:
        generate_data_yaml(
            dataset_root=args.dataset_root,
            output_path=output_path,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo create the dataset structure, run:")
        print(f"  mkdir -p {args.dataset_root}/{{train,val,test}}/{{images,labels}}")


if __name__ == "__main__":
    main()