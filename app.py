"""
Streamlit UI for License Plate Detection and Recognition

Use cases: Smart parking entry/exit, Toll booth automation,
Security and access control systems.
"""

import streamlit as st
import pandas as pd
import cv2
import tempfile
from pathlib import Path

from src.pipeline import PlateRecognitionPipeline


st.set_page_config(page_title="Plate Detector", layout="wide")


def init_pipeline():
    if "pipeline" not in st.session_state:
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
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name)
        tfile.write(uploaded_file.getvalue())
        tfile.close()

        results = pipeline.process_image(tfile.name)

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
                    st.success(f"Plate {i}: **{r['plate_text']}**")
                    st.caption(
                        f"Detection confidence: {r['detection_confidence']:.2%} | "
                        f"OCR confidence: {r['ocr_confidence']:.2%}"
                    )
                    st.caption(f"BBox: {r['bbox']}")
            else:
                st.info("No plates detected in this image.")

        Path(tfile.name).unlink()


def render_video_mode(pipeline):
    st.subheader("Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name)
        tfile.write(uploaded_file.getvalue())
        tfile.close()

        st.info("Processing video... This may take a while for large files.")

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

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

        if results_log:
            st.subheader("Detected Plates During Video")
            df = pd.DataFrame(results_log)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No plates detected in this video.")


def render_webcam_mode(pipeline):
    st.subheader("Live Webcam")

    st.warning("Webcam mode requires an active camera. Make sure your camera is connected.")

    if st.button("Start Webcam"):
        st.info("Starting webcam... Press 'q' in the display window to stop.")

        pipeline.process_video_live(
            video_source=0,
            skip_frames=5,
            dedup_seconds=3.0,
            window_name="Webcam - Press Q to quit",
        )


def render_results_table():
    st.subheader("Detection History")

    csv_path = Path("outputs/results.csv")
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                df_display = df[["timestamp", "plate_text", "detection_confidence", "ocr_confidence"]].copy()
                df_display.columns = ["Timestamp", "Plate Text", "Detection Confidence", "OCR Confidence"]
                df_display = df_display.sort_values("Timestamp", ascending=False).head(50)
                st.dataframe(df_display, use_container_width=True)
            else:
                st.info("No results yet. Process an image or video first.")
        except Exception as e:
            st.error(f"Error reading results: {e}")
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