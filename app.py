import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import tempfile
import os




st.set_page_config(
    page_title="SAR Ship Detection",
    page_icon="🚢",
    layout="wide"
)




MODEL_PATH = "models/best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load model")
    st.code(str(e))
    st.stop()




st.title("🚢 SAR Ship Detection")

st.write(
    "YOLO11s-based ship detection from SAR satellite imagery"
)

st.divider()




st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence",
    0.05,
    0.95,
    0.25,
    0.05
)

imgsz = st.sidebar.selectbox(
    "Image Size",
    [320, 640, 1024],
    index=1
)

mode = st.sidebar.radio(
    "Input Type",
    ["🖼️ Image", "🎥 Video"]
)



if mode == "🖼️ Image":

    uploaded_file = st.file_uploader(
        "Upload SAR Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        results = model.predict(
            image,
            imgsz=imgsz,
            conf=confidence,
            verbose=False
        )

        result = results[0]

        ship_count = len(result.boxes)



        if ship_count > 0:

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            avg_conf = float(
                np.mean(confidences)
            )

        else:

            avg_conf = 0.0




        annotated = result.plot()

        annotated = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )




        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🚢 Ships Detected",
            ship_count
        )

        col2.metric(
            "🎯 Avg Confidence",
            f"{avg_conf:.1%}"
        )

        col3.metric(
            "📐 Image",
            f"{image.width} × {image.height}"
        )


        st.divider()




        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("Detection")

            st.image(
                annotated,
                use_container_width=True
            )



        if ship_count > 0:

            st.divider()

            st.subheader("🚢 Detected Ships")

            data = []

            for i, box in enumerate(result.boxes):

                conf = float(box.conf[0])

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                data.append({
                    "ID": i + 1,
                    "Class": "Ship",
                    "Confidence": f"{conf:.2%}",
                    "X1": round(float(x1), 1),
                    "Y1": round(float(y1), 1),
                    "X2": round(float(x2), 1),
                    "Y2": round(float(y2), 1)
                })

            df = pd.DataFrame(data)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No ships detected. "
                "Try reducing the confidence threshold."
            )




else:

    st.subheader("🎥 SAR Video Detection")

    uploaded_video = st.file_uploader(
        "Upload SAR Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video:


        input_path = os.path.join(
            tempfile.gettempdir(),
            uploaded_video.name
        )

        with open(input_path, "wb") as f:
            f.write(uploaded_video.read())




        output_path = os.path.join(
            tempfile.gettempdir(),
            "sar_ship_detection.mp4"
        )




        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )




        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )


        progress = st.progress(0)

        status = st.empty()

        video_placeholder = st.empty()


        max_count = 0

        frame_number = 0




        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break


            frame_number += 1


            results = model.predict(
                frame,
                imgsz=imgsz,
                conf=confidence,
                verbose=False
            )

            result = results[0]


            

            ship_count = len(
                result.boxes
            )


            max_count = max(
                max_count,
                ship_count
            )


            

            annotated = result.plot()


            # Add ship count

            cv2.putText(
                annotated,
                f"Ships: {ship_count}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3
            )


            writer.write(annotated)


           

            frame_rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            video_placeholder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )


            

            if total_frames > 0:

                progress_value = (
                    frame_number / total_frames
                )

                progress.progress(
                    min(progress_value, 1.0)
                )


            status.write(
                f"Processing frame "
                f"{frame_number}/{total_frames}"
            )




        cap.release()
        writer.release()

        progress.progress(1.0)

        status.success(
            "✅ Video processing completed!"
        )




        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🎞️ Frames",
            frame_number
        )

        col2.metric(
            "🚢 Max Ships / Frame",
            max_count
        )

        col3.metric(
            "⏱️ FPS",
            f"{fps:.1f}"
        )




        if os.path.exists(output_path):

            st.divider()

            st.subheader(
                "📥 Download Result"
            )

            with open(
                output_path,
                "rb"
            ) as f:

                st.download_button(
                    label="⬇️ Download Detection Video",
                    data=f,
                    file_name="SAR_ship_detection.mp4",
                    mime="video/mp4"
                )