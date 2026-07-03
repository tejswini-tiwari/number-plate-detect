"""
Test script for PlateDetector.
"""

import cv2
import sys

sys.path.insert(0, ".")
from src.detector import PlateDetector


def main():
    detector = PlateDetector()

    image_path = "data/images/sample.jpg"
    boxes = detector.detect(image_path)

    image = cv2.imread(image_path)
    result = detector.draw_boxes(image, boxes)

    cv2.imshow("Detection Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()