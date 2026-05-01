import streamlit as st
import streamlit.components.v1 as components
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
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Load CSS from external file
_css_path = os.path.join(os.path.dirname(__file__), "src", "styles.css")
with open(_css_path) as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)


st.markdown("""
<script>
(function() {
  function forwardDrop(e) {
    e.preventDefault();
    var dz = window.parent.document.querySelector(
      '[data-testid="stFileUploaderDropzone"]'
    );
    if (!dz) return;
    // Build a new DataTransfer and copy files across
    var dt = new DataTransfer();
    Array.from(e.dataTransfer.files).forEach(function(f) { dt.items.add(f); });
    var fake = new DragEvent('drop', { bubbles: true, dataTransfer: dt });
    dz.dispatchEvent(fake);
  }
  function forwardOver(e) { e.preventDefault(); }
  // Attach on window so the full page is a drop target
  window.addEventListener('dragover', forwardOver, false);
  window.addEventListener('drop', forwardDrop, false);
  // Also attach directly on the dropzone once it mounts
  function attachDZ() {
    var dz = document.querySelector('[data-testid="stFileUploaderDropzone"]');
    if (dz) {
      dz.addEventListener('dragover', forwardOver, false);
      return true;
    }
    return false;
  }
  if (!attachDZ()) {
    var obs = new MutationObserver(function() { if (attachDZ()) obs.disconnect(); });
    obs.observe(document.body, { childList: true, subtree: true });
  }
})();
</script>
""", unsafe_allow_html=True)

# ── Class metadata ────────────────────────────────────────────────────────────
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
    return load_model(MODEL_PATH, device=DEVICE)

@st.cache_resource
def get_cached_extractor():
    return get_mfcc_extractor()

try:
    model, device = load_cached_model()
    extractor     = get_cached_extractor()
    model_ok      = True
except Exception as _e:
    model_ok  = False
    model_err = str(_e)


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
    type=["wav", "mp3", "flac", "ogg"],
    label_visibility="collapsed",
    accept_multiple_files=False,
)

# PIPELINE
if uploaded_file is not None:

    MAX_BYTES = 50 * 1024 * 1024  # 50 MB limit
    file_bytes = uploaded_file.getvalue()

    if len(file_bytes) == 0:
        st.markdown(
            '<div class="err-box">⚠ The uploaded file is empty. Please try again.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    if len(file_bytes) > MAX_BYTES:
        st.markdown(
            f'<div class="err-box">⚠ File too large ({len(file_bytes)//1024//1024} MB). '
            f'Maximum allowed size is {MAX_BYTES//1024//1024} MB.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.audio(uploaded_file)

    # Write temp file safely
    original_ext = os.path.splitext(uploaded_file.name)[-1].lower() or ".wav"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=original_ext, dir=tempfile.gettempdir()
        ) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name
    except OSError as e:
        st.markdown(
            f'<div class="err-box">⚠ Could not write temporary file: {e}</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # Animated steps
    STEPS = [
        ("Preprocessing audio signal",  "t"),
        ("Removing background noise",   "t"),
        ("Extracting MFCC features",    "p"),
        ("Normalizing feature matrix",  "p"),
        ("Running model inference",     "t"),
    ]
    steps_ph = st.empty()

    def render_steps(n_done: int):
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
        render_steps(0); time.sleep(0.25)
        render_steps(1); time.sleep(0.15)
        render_steps(2); time.sleep(0.15)
        render_steps(3); time.sleep(0.15)
        render_steps(4); time.sleep(0.15)

        pred, probs, elapsed = predict_single_audio(
            temp_path, model, extractor, device,
            target_length=250, return_probs=True,
        )

        steps_ph.empty()

        # Result card
        info       = CLASS_INFO[pred]
        conf       = probs[pred] * 100
        is_norm    = info["badge"] == "normal"
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
                &nbsp;·&nbsp; {info['machine']}
                &nbsp;·&nbsp; {info['status']} state
                &nbsp;·&nbsp; ⏱ {elapsed:.3f}s
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bars
        bars = ""
        for cls_idx in sorted(range(6), key=lambda i: probs[i], reverse=True):
            p       = probs[cls_idx] * 100
            name    = CLASS_INFO[cls_idx]["name"].replace(" — ", " · ")
            fill_cl = "t" if CLASS_INFO[cls_idx]["badge"] == "normal" else "p"
            style   = "font-weight:600;" if cls_idx == pred else "opacity:.55;"
            bars += (
                f'<div class="crow" style="{style}">'
                f'<span class="cname">{name}</span>'
                f'<div class="ctrack">'
                f'<div class="cfill {fill_cl}" style="width:{p:.1f}%"></div>'
                f'</div>'
                f'<span class="cpct">{p:.1f}%</span>'
                f'</div>'
            )
        st.markdown(
            f'<div class="proc-card">'
            f'<div class="proc-title">Confidence — all classes</div>'
            f'{bars}</div>',
            unsafe_allow_html=True,
        )

    except Exception as e:
        steps_ph.empty()
        st.markdown(
            f'<div class="err-box">⚠ Pipeline error: {e}</div>',
            unsafe_allow_html=True,
        )

    finally:
        # Always clean up the temp file, even on error
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass

# FOOTER
st.markdown("""
<div class="footer">
    Pattern Recognition &amp; Neural Networks · Spring 2026<br>
    Built with <span>PyTorch</span> · <span>Librosa</span> · <span>Streamlit</span>
</div>
""", unsafe_allow_html=True)