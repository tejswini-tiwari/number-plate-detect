"""
Streamlit UI for License Plate Detection and Recognition

Use cases: Smart parking entry/exit, Toll booth automation,
Security and access control systems.
"""

import streamlit as st
import pandas as pd
import cv2
import tempfile
import logging
import sys
import os
from pathlib import Path

# Setup logging system for Streamlit
def setup_logging(output_dir: str = "outputs") -> None:
    """Configures the logging system for Streamlit."""
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

setup_logging()
logger = logging.getLogger("streamlit_app")

from src.pipeline import PlateRecognitionPipeline


st.set_page_config(page_title="Plate Detector", layout="wide")


def init_pipeline():
    if "pipeline" not in st.session_state:
        logger.info("Initializing PlateRecognitionPipeline in Streamlit session.")
        st.session_state.pipeline = PlateRecognitionPipeline(
            model_path=None,
            ocr_gpu=False,
            output_dir="outputs",
        )
    return st.session_state.pipeline


def render_header():
    st.title("Vehicle Number Plate Recognition")
    st.markdown("""
    **Smart Parking | Toll Automation | Security & Access Control**

    This system uses YOLO for real-time license plate detection combined with
    OCR (EasyOCR) for text extraction. Upload an image or video, or use your
    webcam to recognize vehicle plates.

    ---""")


def render_sidebar():
    st.sidebar.title("Input Source")
    mode = st.sidebar.radio(
        "Select input type:",
        ["Upload Image", "Upload Video", "Use Webcam"],
        index=0,
    )
    return mode


def render_image_mode(pipeline):
    st.subheader("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded_file is not None:
        logger.info(f"Image uploaded: {uploaded_file.name}")
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name)
        tfile.write(uploaded_file.getvalue())
        tfile.close()

        try:
            results = pipeline.process_image(tfile.name)
        except Exception as e:
            logger.error(f"Error processing uploaded image: {e}", exc_info=True)
            st.error("Failed to process the uploaded image. Check logs.")
            return

        image = cv2.imread(tfile.name)
        annotated = pipeline._draw_annotations(image.copy(), results)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                annotated,
                caption="Annotated Image",
                channels="BGR",
                use_container_width=True,
            )

        with col2:
            st.subheader("Detected Plates")
            if results:
                for i, r in enumerate(results, 1):
                    plate_txt = r['plate_text']
                    if r.get('low_confidence'):
                        st.warning(f"Plate {i} [LOW OCR CONFIDENCE]: **{plate_txt}**")
                        logger.warning(f"Streamlit displayed low-confidence plate {i}: {plate_txt}")
                    else:
                        st.success(f"Plate {i}: **{plate_txt}**")
                        logger.info(f"Streamlit displayed plate {i}: {plate_txt}")
                    st.caption(
                        f"Detection confidence: {r['detection_confidence']:.2%} | "
                        f"OCR confidence: {r['ocr_confidence']:.2%}"
                    )
                    st.caption(f"BBox: {r['bbox']}")
            else:
                st.info("No plates detected in this image.")
                logger.info("Streamlit processing complete. No plates detected.")

        Path(tfile.name).unlink()


def render_video_mode(pipeline):
    st.subheader("Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_file is not None:
        logger.info(f"Video uploaded: {uploaded_file.name}")
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name)
        tfile.write(uploaded_file.getvalue())
        tfile.close()

        st.info("Processing video... This may take a while for large files.")

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            st.error("Failed to parse video frames. Check video encoding.")
            logger.error("Failed to parse video frames. total_frames is 0.")
            return

        frame_window = st.image([])
        progress_bar = st.progress(0)
        status_text = st.empty()

        results_log = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 5 == 0:
                results = pipeline.process_frame(frame)
                for r in results:
                    if r["plate_text"]:
                        results_log.append({
                            "frame": frame_idx,
                            "plate_text": r["plate_text"],
                            "detection_confidence": r["detection_confidence"],
                            "ocr_confidence": r["ocr_confidence"],
                            "low_confidence": r.get("low_confidence", False),
                        })

                annotated = pipeline._draw_annotations(frame.copy(), results)
                frame_window.image(annotated, channels="BGR", use_container_width=True)

            progress = frame_idx / total_frames
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Frame {frame_idx}/{total_frames}")

            frame_idx += 1

        cap.release()
        Path(tfile.name).unlink()

        st.success("Video processing complete!")
        logger.info("Streamlit video processing complete.")

        if results_log:
            st.subheader("Detected Plates During Video")
            df = pd.DataFrame(results_log)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No plates detected in this video.")
            logger.info("No plates detected in the video stream.")


def render_webcam_mode(pipeline):
    st.subheader("Live Webcam")

    st.warning("Webcam mode requires an active camera on the host device. Make sure your camera is connected.")

    if st.button("Start Webcam"):
        st.info("Starting webcam... Press 'q' in the display window to stop.")
        logger.info("Webcam session started by user in Streamlit.")
        try:
            pipeline.process_video_live(
                video_source=0,
                skip_frames=5,
                dedup_seconds=3.0,
                window_name="Webcam - Press Q to quit",
            )
        except Exception as e:
            st.error(f"Error initializing webcam: {e}")
            logger.error(f"Error during Streamlit webcam stream: {e}", exc_info=True)


def render_results_table():
    st.subheader("Detection History")

    csv_path = Path("outputs/results.csv")
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                cols = ["timestamp", "plate_text", "detection_confidence", "ocr_confidence"]
                # Include low_confidence column if it exists
                if "low_confidence" in df.columns:
                    cols.append("low_confidence")
                
                df_display = df[cols].copy()
                
                new_names = ["Timestamp", "Plate Text", "Detection Confidence", "OCR Confidence"]
                if "low_confidence" in df.columns:
                    new_names.append("Low OCR Conf?")
                
                df_display.columns = new_names
                df_display = df_display.sort_values("Timestamp", ascending=False).head(50)
                st.dataframe(df_display, use_container_width=True)
            else:
                st.info("No results yet. Process an image or video first.")
        except Exception as e:
            st.error(f"Error reading results: {e}")
            logger.error(f"Error reading results from CSV in Streamlit: {e}")
    else:
        st.info("No results file found. Process an image or video first.")


def main():
    render_header()
    mode = render_sidebar()
    pipeline = init_pipeline()

    if mode == "Upload Image":
        render_image_mode(pipeline)
    elif mode == "Upload Video":
        render_video_mode(pipeline)
    elif mode == "Use Webcam":
        render_webcam_mode(pipeline)

    st.markdown("---")
    render_results_table()


if __name__ == "__main__":
    main()