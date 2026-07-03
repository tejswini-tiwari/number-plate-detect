"""
Test script for PlateOCR.
"""

import cv2
import sys

sys.path.insert(0, ".")
from src.ocr_reader import PlateOCR


def main():
    ocr = PlateOCR(gpu=False)

    image_path = "data/images/sample_plate.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image from {image_path}")
        print("Place a cropped plate image at data/images/sample_plate.jpg")
        return

    text, conf = ocr.read_plate(image)

    print(f"Recognized Text: {text}")
    print(f"Confidence: {conf:.2%}")


if __name__ == "__main__":
    main()