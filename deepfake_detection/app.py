"""
TruthLens AI - Deepfake Detection

Current trained model:
    trained_models/deepfake_image_model.pth

Model:
    EfficientNet-B0

Classes:
    0 -> fake
    1 -> real

Image preprocessing is kept identical to predict_image.py:
    RGB -> Resize 224x224 -> Tensor -> ImageNet Normalize

Video detection:
    Video -> Sample frames -> Same trained image model
           -> Frame-level probabilities
           -> Robust aggregation
           -> Final result
"""

import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
import timm

from PIL import Image
from torchvision import transforms


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TruthLens AI - Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "trained_models",
    "deepfake_image_model.pth",
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading trained deepfake model...")
def load_trained_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Trained model not found:\n"
            f"{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    classes = checkpoint["classes"]

    # Safety check
    if classes != ["fake", "real"]:

        raise ValueError(
            "Unexpected class order in checkpoint: "
            f"{classes}. Expected ['fake', 'real']."
        )

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=2
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    return model, classes, checkpoint


try:

    model, classes, checkpoint = load_trained_model()

    MODEL_READY = True

except Exception as e:

    model = None
    classes = ["fake", "real"]
    checkpoint = None
    MODEL_READY = False
    MODEL_ERROR = str(e)


# ============================================================
# IMAGE PREPROCESSING
# EXACTLY MATCHES predict_image.py
# ============================================================

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# MODEL PREDICTION FOR PIL IMAGE
# ============================================================

def predict_pil_image(image):

    if not MODEL_READY:

        raise RuntimeError(
            MODEL_ERROR
        )

    image = image.convert("RGB")

    tensor = IMAGE_TRANSFORM(
        image
    ).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.inference_mode():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    fake_probability = float(
        probabilities[0].item()
    )

    real_probability = float(
        probabilities[1].item()
    )

    if fake_probability >= real_probability:

        prediction = "FAKE"

        confidence = fake_probability

    else:

        prediction = "REAL"

        confidence = real_probability

    return {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
    }


# ============================================================
# MODEL PREDICTION FOR OPENCV FRAME
# ============================================================

def predict_frame(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    return predict_pil_image(
        image
    )


# ============================================================
# ROBUST VIDEO AGGREGATION
# ============================================================

def aggregate_video_predictions(
    fake_probabilities
):

    values = np.asarray(
        fake_probabilities,
        dtype=np.float32
    )

    if len(values) == 0:

        raise ValueError(
            "No frame probabilities available."
        )

    # Remove extreme outliers when enough
    # frames are available.
    if len(values) >= 10:

        lower = np.percentile(
            values,
            10
        )

        upper = np.percentile(
            values,
            90
        )

        trimmed = values[
            (values >= lower)
            &
            (values <= upper)
        ]

        if len(trimmed) >= 5:

            values = trimmed

    # Median is more robust than simple mean
    # against a few unusual frames.
    fake_probability = float(
        np.median(values)
    )

    real_probability = (
        1.0 -
        fake_probability
    )

    # Avoid pretending that near-50/50
    # predictions are highly certain.
    margin = abs(
        fake_probability -
        real_probability
    )

    if margin < 0.10:

        prediction = "UNCERTAIN"

    elif fake_probability > real_probability:

        prediction = "FAKE"

    else:

        prediction = "REAL"

    confidence = max(
        fake_probability,
        real_probability
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "frames_used_for_aggregation": len(values),
    }


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(
    video_path,
    max_frames=30,
    progress_callback=None
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise ValueError(
            "Unable to open the uploaded video."
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if total_frames <= 0:

        cap.release()

        raise ValueError(
            "The video contains no readable frames."
        )

    # --------------------------------------------------------
    # EVENLY SAMPLE FRAMES
    # --------------------------------------------------------

    sample_count = min(
        max_frames,
        total_frames
    )

    if total_frames <= sample_count:

        frame_indices = np.arange(
            total_frames
        )

    else:

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            sample_count,
            dtype=np.int64
        )

    fake_probabilities = []

    real_probabilities = []

    frame_predictions = []

    frame_numbers = []

    processed_frames = []

    # --------------------------------------------------------
    # PROCESS FRAMES
    # --------------------------------------------------------

    for current_index, frame_index in enumerate(
        frame_indices,
        start=1
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = cap.read()

        if not success:
            continue

        try:

            result = predict_frame(
                frame
            )

        except Exception:
            continue

        fake_probability = result[
            "fake_probability"
        ]

        real_probability = result[
            "real_probability"
        ]

        prediction = result[
            "prediction"
        ]

        fake_probabilities.append(
            fake_probability
        )

        real_probabilities.append(
            real_probability
        )

        frame_predictions.append(
            prediction
        )

        frame_numbers.append(
            int(frame_index)
        )

        processed_frames.append(
            {
                "frame":
                    int(frame_index),

                "prediction":
                    prediction,

                "fake_probability":
                    fake_probability,

                "real_probability":
                    real_probability,
            }
        )

        if progress_callback:

            progress_callback(
                current_index /
                len(frame_indices)
            )

    cap.release()

    if len(fake_probabilities) == 0:

        raise ValueError(
            "No video frames could be analyzed."
        )

    # --------------------------------------------------------
    # AGGREGATE
    # --------------------------------------------------------

    final_result = aggregate_video_predictions(
        fake_probabilities
    )

    # --------------------------------------------------------
    # ADD VIDEO METADATA
    # --------------------------------------------------------

    final_result[
        "total_frames"
    ] = total_frames

    final_result[
        "fps"
    ] = fps

    final_result[
        "duration_seconds"
    ] = (
        total_frames / fps
        if fps > 0
        else 0.0
    )

    final_result[
        "frames_analyzed"
    ] = len(
        processed_frames
    )

    final_result[
        "frame_results"
    ] = processed_frames

    final_result[
        "fake_frame_count"
    ] = sum(
        p == "FAKE"
        for p in frame_predictions
    )

    final_result[
        "real_frame_count"
    ] = sum(
        p == "REAL"
        for p in frame_predictions
    )

    final_result[
        "uncertain_frame_count"
    ] = sum(
        p == "UNCERTAIN"
        for p in frame_predictions
    )

    return final_result


# ============================================================
# UI HELPERS
# ============================================================

def show_result(
    prediction,
    confidence,
    real_probability,
    fake_probability
):

    st.markdown("---")

    st.subheader(
        "Detection Result"
    )

    if prediction == "FAKE":

        st.error(
            "🚨 Prediction: **DEEPFAKE**"
        )

    elif prediction == "REAL":

        st.success(
            "✅ Prediction: **REAL**"
        )

    else:

        st.warning(
            "⚠️ Prediction: **UNCERTAIN**"
        )

        st.caption(
            "The model's evidence is too close "
            "to confidently classify this input."
        )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Real Probability",
        f"{real_probability * 100:.2f}%"
    )

    col2.metric(
        "Deepfake Probability",
        f"{fake_probability * 100:.2f}%"
    )

    col3.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )


def create_probability_chart(
    real_probability,
    fake_probability
):

    import plotly.graph_objects as go

    figure = go.Figure(
        data=[
            go.Bar(
                x=[
                    real_probability * 100,
                    fake_probability * 100
                ],
                y=[
                    "REAL",
                    "DEEPFAKE"
                ],
                orientation="h",
                text=[
                    f"{real_probability * 100:.2f}%",
                    f"{fake_probability * 100:.2f}%"
                ],
                textposition="auto",
            )
        ]
    )

    figure.update_layout(
        title="Prediction Probability",
        xaxis_title="Probability (%)",
        xaxis=dict(
            range=[0, 100]
        ),
        height=300,
    )

    return figure


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🛡️ TruthLens AI"
)

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🖼️ Image Detection",
        "🎥 Video Detection",
        "📊 Model Information",
        "ℹ️ About Project",
    ],
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"**Compute device:** {DEVICE.type.upper()}"
)

if MODEL_READY:

    st.sidebar.success(
        "Trained Image Model: READY"
    )

    if checkpoint is not None:

        best_val = checkpoint.get(
            "val_accuracy"
        )

        if best_val is not None:

            st.sidebar.write(
                f"Best validation accuracy: "
                f"{best_val * 100:.2f}%"
            )

else:

    st.sidebar.error(
        "Trained model could not be loaded."
    )


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title(
        "🛡️ TruthLens AI"
    )

    st.subheader(
        "Deepfake Image & Video Detection"
    )

    st.markdown(
        """
        Upload an image or video and the trained
        EfficientNet-B0 deepfake detector will
        analyze it.

        **Current trained pipeline**

        Video → Frame Sampling → EfficientNet-B0
        → Frame Predictions → Robust Aggregation
        → Final Result
        """
    )

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Model",
            "EfficientNet-B0"
        )

    with c2:

        st.metric(
            "Test Accuracy",
            "97.47%"
        )

    with c3:

        st.metric(
            "Best Validation",
            "97.74%"
        )

    st.info(
        "The current trained model is an image/frame "
        "classifier. Video detection applies this "
        "trained model to multiple sampled video frames."
    )


# ============================================================
# IMAGE DETECTION
# ============================================================

elif page == "🖼️ Image Detection":

    st.title(
        "🖼️ Image Deepfake Detection"
    )

    st.write(
        "The uploaded image is processed using "
        "the same preprocessing used during model testing."
    )

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="image_uploader"
    )

    if uploaded_image is not None:

        image = Image.open(
            uploaded_image
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button(
            "🔍 Detect Image",
            type="primary"
        ):

            if not MODEL_READY:

                st.error(
                    MODEL_ERROR
                )

                st.stop()

            with st.spinner(
                "Analyzing image..."
            ):

                try:

                    result = predict_pil_image(
                        image
                    )

                except Exception as e:

                    st.error(
                        f"Prediction failed: {e}"
                    )

                    st.stop()

            show_result(
                result["prediction"],
                result["confidence"],
                result["real_probability"],
                result["fake_probability"]
            )

            st.plotly_chart(
                create_probability_chart(
                    result["real_probability"],
                    result["fake_probability"]
                ),
                use_container_width=True
            )


# ============================================================
# VIDEO DETECTION
# ============================================================

elif page == "🎥 Video Detection":

    st.title(
        "🎥 Video Deepfake Detection"
    )

    st.write(
        "The trained image model analyzes multiple "
        "sampled frames from the uploaded video."
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ],
        key="video_uploader"
    )

    max_frames = st.slider(
        "Frames to analyze",
        min_value=10,
        max_value=60,
        value=30,
        step=5
    )

    if uploaded_video is not None:

        st.video(
            uploaded_video
        )

        if st.button(
            "🔍 Analyze Video",
            type="primary"
        ):

            temp_path = None

            try:

                suffix = os.path.splitext(
                    uploaded_video.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temporary_file:

                    temporary_file.write(
                        uploaded_video.getbuffer()
                    )

                    temp_path = (
                        temporary_file.name
                    )

                progress_bar = st.progress(
                    0
                )

                status = st.empty()

                def update_progress(value):

                    progress_bar.progress(
                        min(
                            max(
                                int(value * 100),
                                0
                            ),
                            100
                        )
                    )

                    status.write(
                        f"Analyzing video frames... "
                        f"{value * 100:.0f}%"
                    )

                with st.spinner(
                    "Starting video analysis..."
                ):

                    result = analyze_video(
                        temp_path,
                        max_frames=max_frames,
                        progress_callback=update_progress
                    )

                progress_bar.progress(
                    100
                )

                status.success(
                    "Video analysis completed."
                )

                # ------------------------------------------
                # RESULT
                # ------------------------------------------

                show_result(
                    result["prediction"],
                    result["confidence"],
                    result["real_probability"],
                    result["fake_probability"]
                )

                # ------------------------------------------
                # VIDEO INFORMATION
                # ------------------------------------------

                st.markdown(
                    "### Video Analysis"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Frames Analyzed",
                    result["frames_analyzed"]
                )

                c2.metric(
                    "Total Frames",
                    result["total_frames"]
                )

                c3.metric(
                    "FPS",
                    f"{result['fps']:.2f}"
                )

                c4.metric(
                    "Duration",
                    f"{result['duration_seconds']:.2f}s"
                )

                # ------------------------------------------
                # FRAME COUNTS
                # ------------------------------------------

                st.markdown(
                    "### Frame-Level Summary"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "FAKE Frames",
                    result["fake_frame_count"]
                )

                c2.metric(
                    "REAL Frames",
                    result["real_frame_count"]
                )

                c3.metric(
                    "Uncertain Frames",
                    result["uncertain_frame_count"]
                )

                # ------------------------------------------
                # PROBABILITY CHART
                # ------------------------------------------

                st.plotly_chart(
                    create_probability_chart(
                        result["real_probability"],
                        result["fake_probability"]
                    ),
                    use_container_width=True
                )

                # ------------------------------------------
                # FRAME PROBABILITY TABLE
                # ------------------------------------------

                st.markdown(
                    "### Per-Frame Deepfake Probability"
                )

                frame_results = result[
                    "frame_results"
                ]

                if frame_results:

                    st.dataframe(
                        [
                            {
                                "Frame":
                                    item["frame"],

                                "Prediction":
                                    item["prediction"],

                                "Fake Probability":
                                    f"{item['fake_probability'] * 100:.2f}%",

                                "Real Probability":
                                    f"{item['real_probability'] * 100:.2f}%"
                            }

                            for item in frame_results
                        ],
                        use_container_width=True
                    )

            except Exception as e:

                st.error(
                    f"Video analysis failed: {e}"
                )

            finally:

                if (
                    temp_path
                    and
                    os.path.exists(temp_path)
                ):

                    try:

                        os.remove(
                            temp_path
                        )

                    except OSError:

                        pass


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "📊 Model Information":

    st.title(
        "📊 Model Information"
    )

    st.markdown(
        "### Trained Model"
    )

    st.code(
        """
Input Image
    ↓
Resize 224 × 224
    ↓
ImageNet Normalization
    ↓
EfficientNet-B0
    ↓
2-Class Classifier
    ↓
FAKE / REAL
        """,
        language="text"
    )

    st.markdown(
        "### Class Mapping"
    )

    st.code(
        """
Class 0 → FAKE
Class 1 → REAL
        """,
        language="text"
    )

    st.markdown(
        "### Training Results"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Training Images",
        "42,000"
    )

    c2.metric(
        "Validation Images",
        "9,000"
    )

    c3.metric(
        "Test Images",
        "9,000"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Best Validation Accuracy",
        "97.74%"
    )

    c2.metric(
        "Test Accuracy",
        "97.47%"
    )

    st.warning(
        "These accuracy figures come from the prepared "
        "test/validation splits. Because the dataset contains "
        "video-derived frames, a video-level cross-source "
        "evaluation is still needed to estimate generalization "
        "to completely unseen videos."
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.title(
        "ℹ️ About TruthLens AI"
    )

    st.markdown(
        """
        TruthLens AI is a deepfake detection project that
        uses a trained EfficientNet-B0 image classifier.

        The current implementation analyzes individual images
        and multiple sampled frames from videos.
        """
    )

    st.markdown(
        "### Current Pipeline"
    )

    st.code(
        """
Image:
Image
 ↓
Preprocessing
 ↓
EfficientNet-B0
 ↓
FAKE / REAL


Video:
Video
 ↓
Frame Sampling
 ↓
EfficientNet-B0 per Frame
 ↓
Frame Probabilities
 ↓
Robust Aggregation
 ↓
FINAL RESULT
        """,
        language="text"
    )

    st.warning(
        "This system is a decision-support tool. "
        "A prediction is not a guarantee of authenticity, "
        "and real-world performance can differ from the "
        "held-out test accuracy."
    )