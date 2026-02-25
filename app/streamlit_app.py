"""
DeepFake Detection — Streamlit Web Application.

A modern, production-ready web interface for detecting deepfakes
in images and videos using an EfficientNet-B4 classifier.

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
import logging
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import Config
from models.detector import DeepFakeDetector
from preprocessing.face_detector import FaceDetector
from preprocessing.frame_extractor import FrameExtractor
from preprocessing.data_pipeline import get_inference_transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg-primary: #0f0f23;
        --bg-secondary: #1a1a3e;
        --bg-card: rgba(26, 26, 62, 0.7);
        --accent-blue: #667eea;
        --accent-purple: #764ba2;
        --accent-green: #00d2ff;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --real-color: #10b981;
        --fake-color: #ef4444;
        --border-color: rgba(102, 126, 234, 0.2);
    }

    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 300;
    }

    .result-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }

    .result-real {
        border-color: var(--real-color);
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.15);
    }
    .result-fake {
        border-color: var(--fake-color);
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.15);
    }

    .result-label {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .label-real { color: var(--real-color); }
    .label-fake { color: var(--fake-color); }

    .confidence-text {
        font-size: 1.3rem;
        font-weight: 500;
        color: var(--text-secondary);
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    .metric-item {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--accent-blue);
    }
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 0.3rem;
    }

    .frame-analysis {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .sidebar .sidebar-content {
        background: var(--bg-secondary);
    }

    .upload-area {
        background: var(--bg-card);
        border: 2px dashed var(--border-color);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached Model Loading ──────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the deepfake detection model (cached)."""
    config = Config()
    model_path = config.model_save_path

    if not model_path.exists():
        return None, config

    try:
        model = DeepFakeDetector.load_from_checkpoint(
            str(model_path), device=config.device
        )
        return model, config
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, config


@st.cache_resource
def load_face_detector():
    """Load MTCNN face detector (cached)."""
    return FaceDetector(confidence_threshold=0.9)


@st.cache_resource
def load_frame_extractor():
    """Load frame extractor (cached)."""
    config = Config()
    return FrameExtractor(
        frame_interval=config.frame_interval,
        max_frames=config.max_frames,
    )


# ── Main App ──────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 DeepFake Detector</h1>
        <p>AI-powered deepfake detection using EfficientNet-B4</p>
    </div>
    """, unsafe_allow_html=True)

    # Load resources
    model, config = load_model()
    face_detector = load_face_detector()
    frame_extractor = load_frame_extractor()
    transform = get_inference_transform(config.image_size)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")

        st.markdown("**Model Info**")
        if model is not None:
            st.success("✅ Model loaded")
            st.markdown(f"- Architecture: `{config.model_name}`")
            st.markdown(f"- Parameters: `{model.get_total_params():,}`")
            st.markdown(f"- Device: `{config.device}`")
        else:
            st.warning("⚠️ No model checkpoint found")
            st.markdown(
                "Train the model first:\n"
                "```bash\n"
                "python -m training.train\n"
                "```"
            )

        st.markdown("---")
        st.markdown("**About**")
        st.markdown(
            "This system uses an **EfficientNet-B4** CNN "
            "fine-tuned for binary deepfake classification. "
            "Upload an image or video to get started."
        )

        st.markdown("---")
        st.markdown("**Video Settings**")
        frame_interval = st.slider("Frame interval", 5, 30, config.frame_interval)
        max_frames = st.slider("Max frames", 10, 100, config.max_frames)

    # Main content
    tab1, tab2 = st.tabs(["📷 Image Detection", "🎬 Video Detection"])

    # ── Image Tab ──────────────────────────────────────────────
    with tab1:
        st.markdown("### Upload an image to analyze")
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="image_upload",
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)

            with col2:
                if model is None:
                    st.error(
                        "❌ No model loaded. Please train the model first.\n\n"
                        "```bash\npython -m training.train --data_dir data/processed\n```"
                    )
                else:
                    with st.spinner("🔍 Analyzing image..."):
                        # Detect and crop face
                        face_image = face_detector.detect_and_crop(
                            image, target_size=config.image_size
                        )

                        # Predict
                        label, confidence, probs = model.predict_image(
                            face_image, transform, config.device
                        )

                    # Display result
                    css_class = "result-real" if label == "REAL" else "result-fake"
                    label_class = "label-real" if label == "REAL" else "label-fake"
                    emoji = "✅" if label == "REAL" else "🚨"

                    st.markdown(f"""
                    <div class="result-card {css_class}">
                        <div class="result-label {label_class}">{emoji} {label}</div>
                        <div class="confidence-text">Confidence: {confidence:.1%}</div>
                        <div class="metrics-grid">
                            <div class="metric-item">
                                <div class="metric-value">{probs['REAL']:.1%}</div>
                                <div class="metric-label">Real Probability</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-value">{probs['FAKE']:.1%}</div>
                                <div class="metric-label">Fake Probability</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show cropped face
                    if face_image is not None:
                        st.markdown("**Detected Face Region**")
                        st.image(face_image, width=200)

    # ── Video Tab ──────────────────────────────────────────────
    with tab2:
        st.markdown("### Upload a video to analyze")
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_upload",
        )

        if uploaded_video is not None:
            st.video(uploaded_video)

            if model is None:
                st.error(
                    "❌ No model loaded. Please train the model first.\n\n"
                    "```bash\npython -m training.train --data_dir data/processed\n```"
                )
            else:
                if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                    # Extract frames
                    with st.spinner("📼 Extracting frames..."):
                        frame_extractor_instance = FrameExtractor(
                            frame_interval=frame_interval,
                            max_frames=max_frames,
                        )
                        frames = frame_extractor_instance.extract_frames_from_bytes(
                            uploaded_video.getvalue()
                        )

                    st.info(f"Extracted **{len(frames)}** frames for analysis")

                    # Process frames
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    face_frames = []
                    with st.spinner("👤 Detecting faces..."):
                        for i, frame in enumerate(frames):
                            face = face_detector.detect_and_crop(
                                frame, target_size=config.image_size
                            )
                            face_frames.append(face)
                            progress_bar.progress((i + 1) / len(frames))
                            status_text.text(f"Processing frame {i + 1}/{len(frames)}")

                    status_text.empty()
                    progress_bar.empty()

                    # Predict on all frames
                    with st.spinner("🧠 Running inference..."):
                        overall_label, overall_conf, frame_results = model.predict_video(
                            face_frames, transform, config.device
                        )

                    # Display overall result
                    css_class = "result-real" if overall_label == "REAL" else "result-fake"
                    label_class = "label-real" if overall_label == "REAL" else "label-fake"
                    emoji = "✅" if overall_label == "REAL" else "🚨"

                    fake_count = sum(1 for r in frame_results if r["label"] == "FAKE")
                    real_count = len(frame_results) - fake_count

                    st.markdown(f"""
                    <div class="result-card {css_class}">
                        <div class="result-label {label_class}">{emoji} {overall_label}</div>
                        <div class="confidence-text">Overall Confidence: {overall_conf:.1%}</div>
                        <div class="metrics-grid">
                            <div class="metric-item">
                                <div class="metric-value">{len(frame_results)}</div>
                                <div class="metric-label">Frames Analyzed</div>
                            </div>
                            <div class="metric-item">
                                <div class="metric-value">{fake_count}/{real_count}</div>
                                <div class="metric-label">Fake / Real Frames</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Per-frame analysis
                    with st.expander("📊 Per-Frame Analysis", expanded=False):
                        cols = st.columns(5)
                        for i, result in enumerate(frame_results):
                            col = cols[i % 5]
                            with col:
                                color = "#10b981" if result["label"] == "REAL" else "#ef4444"
                                st.markdown(f"""
                                <div class="frame-analysis">
                                    <div style="font-weight:600; font-size:0.85rem;">
                                        Frame {result['frame_idx'] + 1}
                                    </div>
                                    <div style="color:{color}; font-weight:700; font-size:1.1rem;">
                                        {result['label']}
                                    </div>
                                    <div style="color:var(--text-secondary); font-size:0.8rem;">
                                        {result['confidence']:.1%}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
