import streamlit as st
import torch
import os
import tempfile
import time

from src.train.inference_utils import (
    load_model, 
    get_mfcc_extractor,
    predict_single_audio
)

# Page config
st.set_page_config(
    page_title="Machine Fault Recognition",
    page_icon="🔊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Load CSS from external file
_css_path = os.path.join(os.path.dirname(__file__), "src", "styles.css")
with open(_css_path) as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)


# Class metadata
CLASS_INFO = {
    0: {"name": "Machine 1 — Normal",   "machine": "Machine 1", "status": "Normal",   "badge": "normal"},
    1: {"name": "Machine 1 — Abnormal", "machine": "Machine 1", "status": "Abnormal", "badge": "abnormal"},
    2: {"name": "Machine 2 — Normal",   "machine": "Machine 2", "status": "Normal",   "badge": "normal"},
    3: {"name": "Machine 2 — Abnormal", "machine": "Machine 2", "status": "Abnormal", "badge": "abnormal"},
    4: {"name": "Machine 3 — Normal",   "machine": "Machine 3", "status": "Normal",   "badge": "normal"},
    5: {"name": "Machine 3 — Abnormal", "machine": "Machine 3", "status": "Abnormal", "badge": "abnormal"},
}

# Load model and feature extractor with caching
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'src/model/model_epoch_75.pkl'

@st.cache_resource
def load_cached_model():
    model, device = load_model(MODEL_PATH, device=DEVICE)
    return model, device

@st.cache_resource
def get_cached_extractor():
    return get_mfcc_extractor()

try:
    model, device = load_cached_model()
    extractor = get_cached_extractor()
    model_ok  = True
except Exception as e:
    model_ok  = False
    model_err = str(e)


# HERO
st.markdown("""
<div class="hero">
    <div class="hero-badge">Pattern Recognition · Spring 2026</div>
    <h1 class="hero-title">
        Machine <span class="t">Fault</span><br><span class="p">Recognition</span>
    </h1>
    <span class="hero-sub">
        Upload an audio recording to detect whether a machine is<br>operating normally or experiencing a fault condition.
    </span>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

if not model_ok:
    st.markdown(f'<div class="err-box">⚠ Model failed to load: {model_err}</div>',
                unsafe_allow_html=True)

# DROP ZONE
uploaded_file = st.file_uploader(
    label="Upload Audio",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    label_visibility="collapsed",
)

# PIPELINE
if uploaded_file is not None:

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    STEPS = [
        ("Preprocessing audio signal",  "t"),
        ("Removing background noise",   "t"),
        ("Extracting MFCC features",    "p"),
        ("Normalizing feature matrix",  "p"),
        ("Running model inference",     "t"),
    ]
    steps_ph = st.empty()

    def render_steps(n_done):
        rows = ""
        for i, (label, color) in enumerate(STEPS):
            if i < n_done:
                rows += (f'<div class="step-row done">'
                         f'<div class="sdot {color}"></div>{label} ✓</div>')
            elif i == n_done:
                rows += (f'<div class="step-row">'
                         f'<div class="sdot {color} pulse"></div>{label} …</div>')
        steps_ph.markdown(
            f'<div class="proc-card"><div class="proc-title">Processing pipeline</div>{rows}</div>',
            unsafe_allow_html=True,
        )

    try:
        # Step 0: preprocess (timer starts AFTER file read)
        render_steps(0); time.sleep(0.3)

        # Run inference (all steps done inside predict_single_audio)
        render_steps(1); time.sleep(0.2)
        render_steps(2); time.sleep(0.2)
        render_steps(3); time.sleep(0.2)
        render_steps(4); time.sleep(0.25)
        
        pred, probs, elapsed = predict_single_audio(
            temp_path, model, extractor, device, 
            target_length=250, return_probs=True
        )

        steps_ph.empty()
        os.unlink(temp_path)

        # Result 
        info    = CLASS_INFO[pred]
        conf    = probs[pred] * 100
        is_norm = info["badge"] == "normal"
        badge_html = ('<span class="badge-ok">● Normal Operation</span>'
                      if is_norm else
                      '<span class="badge-err">⚠ Fault Detected</span>')
        conf_color = "teal" if is_norm else "pink"

        st.markdown(f"""
        <div class="result-card">
            <div class="rl">Prediction Result</div>
            <div class="rn">{pred}</div>
            <div class="rnm">{info['name']}</div>
            {badge_html}
            <div class="rmeta">
                Confidence:
                <strong class="{conf_color}">{conf:.1f}%</strong>
                &nbsp;·&nbsp; {info['machine']} &nbsp;·&nbsp; {info['status']} state
                &nbsp;·&nbsp; ⏱ {elapsed:.3f}s
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence breakdown
        bars = ""
        for cls_idx in sorted(range(6), key=lambda i: probs[i], reverse=True):
            p       = probs[cls_idx] * 100
            name    = CLASS_INFO[cls_idx]["name"].replace(" — ", " · ")
            fill_cl = "t" if CLASS_INFO[cls_idx]["badge"] == "normal" else "p"
            bold    = "font-weight:600;" if cls_idx == pred else "opacity:.55;"
            bars += (f'<div class="crow" style="{bold}">'
                     f'<span class="cname">{name}</span>'
                     f'<div class="ctrack"><div class="cfill {fill_cl}" style="width:{p:.1f}%"></div></div>'
                     f'<span class="cpct">{p:.1f}%</span></div>')

        st.markdown(
            f'<div class="proc-card"><div class="proc-title">Confidence — all classes</div>{bars}</div>',
            unsafe_allow_html=True,
        )

    except Exception as e:
        steps_ph.empty()
        try: os.unlink(temp_path)
        except: pass
        st.markdown(f'<div class="err-box">⚠ Pipeline error: {e}</div>',
                    unsafe_allow_html=True)


# FOOTER
st.markdown("""
<div class="footer">
    Pattern Recognition & Neural Networks · Spring 2026<br>
    Built with <span>PyTorch</span> · <span>Librosa</span> · <span>Streamlit</span>
</div>
""", unsafe_allow_html=True)