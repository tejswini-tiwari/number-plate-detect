# Plate Detector

Real-time vehicle number plate detection and recognition using YOLO and OCR.

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO weights (place in models/ directory)
```

## Project Structure

```
plate-detector/
├── src/
│   ├── detector.py       # YOLO detection logic
│   ├── ocr_reader.py     # OCR extraction logic
│   ├── pipeline.py       # combines detection + OCR
│   └── utils.py          # preprocessing helpers
├── data/
│   ├── images/
│   └── videos/
├── models/               # YOLO weights
├── outputs/              # detected results, cropped plates, logs
├── requirements.txt
├── main.py
└── README.md
```

## Usage

```bash
python main.py
```