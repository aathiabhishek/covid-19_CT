import hashlib
import re
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="LungLens CT", page_icon="🫁", layout="wide")

# ----------------------------------------------------------------- settings
CLASSES = {
    "nCT": {"title": "Normal", "verdict": "No signs of COVID-19 found",
            "detail": "The lungs are visible and show no changes typical of COVID-19 pneumonia.",
            "tone": "ok", "count": 9979},
    "pCT": {"title": "Positive", "verdict": "Changes typical of COVID-19 pneumonia",
            "detail": "The lungs show patterns that are commonly seen in COVID-19 pneumonia.",
            "tone": "alert", "count": 4001},
    "NiCT": {"title": "Non-informative", "verdict": "This slice can't be assessed",
             "detail": "The slice does not show enough of the lungs to judge. Try a slice taken through the middle of the chest.",
             "tone": "neutral", "count": 5705},
}
LOW_CONFIDENCE = 0.60

# -------------------------------------------------------------- your model
@st.cache_resource
@st.cache_resource
def load_model():
    import torch
    from torch import nn
    from torchvision import models

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(960, 1280), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1280, 3)
            )

        def forward(self, x):
            return self.net(x)

    checkpoint = Path(__file__).parent / "outputs/checkpoints/mobilenet_v3_large.pth"
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = models.mobilenet_v3_large(weights=None)
    model.classifier = Head()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict(img: Image.Image, raw_bytes: bytes):
    """Return ({"nCT": p, "pCT": p, "NiCT": p}, is_demo)."""
    model = load_model()
    if model is None:  # placeholder numbers so you can preview the UI
        seed = int(hashlib.md5(raw_bytes).hexdigest()[:8], 16)
        p = np.random.default_rng(seed).dirichlet([2.0, 1.2, 1.2])
        return {"nCT": float(p[0]), "pCT": float(p[1]), "NiCT": float(p[2])}, True

    import torch
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = preprocess(img.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probs = torch.softmax(model(image), dim=1)[0].tolist()
    # Checkpoint class indices: NiCT=0, nCT=1, pCT=2.
    return {"nCT": float(probs[1]), "pCT": float(probs[2]), "NiCT": float(probs[0])}, False

# ----------------------------------------------------------------- helpers
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>", unsafe_allow_html=True)


def html(s: str):
    st.markdown(re.sub(r"^\s+", "", s, flags=re.M), unsafe_allow_html=True)


LOGO = """<svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect width="32" height="32" rx="8" fill="#0f6b73"/>
<path d="M11 8c-3 2-5 8-5 13 0 2 1 3 3 3 2 0 3-1 3-3V8zM21 8c3 2 5 8 5 13 0 2-1 3-3 3-2 0-3-1-3-3V8z" fill="#fff"/>
</svg>"""

LUNG = """<svg viewBox="0 0 320 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Illustration of a chest CT slice">
<defs><filter id="b"><feGaussianBlur stdDeviation="4"/></filter></defs>
<circle cx="160" cy="160" r="150" fill="#0d2329"/>
<ellipse cx="160" cy="165" rx="125" ry="105" fill="#17363d"/>
<ellipse cx="110" cy="165" rx="50" ry="78" fill="#cfe3e6" opacity=".88"/>
<ellipse cx="210" cy="165" rx="50" ry="78" fill="#cfe3e6" opacity=".88"/>
<ellipse cx="88" cy="150" rx="14" ry="20" fill="#fff" opacity=".75" filter="url(#b)"/>
<ellipse cx="232" cy="185" rx="16" ry="22" fill="#fff" opacity=".7" filter="url(#b)"/>
<ellipse cx="160" cy="215" rx="14" ry="20" fill="#a7c4c9"/>
<ellipse cx="160" cy="150" rx="12" ry="26" fill="#2c525a"/>
</svg>"""

# -------------------------------------------------------------- home page
def home():
    c = CLASSES
    html(f"""
    <div class="hero">
    <div>
    <h1>Check a chest CT slice for signs of COVID-19 pneumonia</h1>
    <p>Upload a CT image and get a result in seconds. The classifier tells you whether the slice looks normal, shows changes typical of COVID-19, or can't be judged.</p>
    </div>
    {LUNG}
    </div>
    """)
    st.write("")
    st.page_link(analyze_page, label="Analyze a scan", icon=":material/upload:")

    html("""
    <div class="sec">
    <h2>About COVID-19</h2>
    <p class="lead">COVID-19 is an infectious disease caused by the SARS-CoV-2 virus. It mainly affects the respiratory system, and in some people it leads to pneumonia.</p>
    </div>
    <div class="two">
    <div>
    <h3>What it does to the lungs</h3>
    <p>In COVID-19 pneumonia, small air sacs in the lungs fill with fluid and inflammation. On a CT image this often appears as hazy grey patches called ground-glass opacities, usually in both lungs and near the outer edges.</p>
    <h3>Common symptoms</h3>
    <p>Fever, cough, shortness of breath, fatigue and loss of taste or smell. Some people have mild symptoms or none at all.</p>
    </div>
    <div>
    <h3>Why CT imaging is used</h3>
    <p>A CT scan shows the lungs in thin slices, so doctors can see how much of the lung is affected. It helps assess severity and track recovery.</p>
    <h3>What CT can't do</h3>
    <p>CT is not the standard test for diagnosing COVID-19. Other lung conditions can look similar, and a normal scan does not rule out infection. Diagnosis relies on laboratory tests such as RT-PCR, together with symptoms and clinical assessment.</p>
    </div>
    </div>
    """)

    html(f"""
    <div class="sec">
    <h2>What the classifier reports</h2>
    <p class="lead">Each slice is sorted into one of three groups. The model was trained on {sum(v['count'] for v in c.values()):,} labelled CT images.</p>
    </div>
    <div class="classes">
    <div class="cls ok"><h3>{c['nCT']['title']} <code>nCT</code></h3><p>{c['nCT']['detail']}</p><span class="n">{c['nCT']['count']:,} training images</span></div>
    <div class="cls alert"><h3>{c['pCT']['title']} <code>pCT</code></h3><p>{c['pCT']['detail']}</p><span class="n">{c['pCT']['count']:,} training images</span></div>
    <div class="cls neutral"><h3>{c['NiCT']['title']} <code>NiCT</code></h3><p>{c['NiCT']['detail']}</p><span class="n">{c['NiCT']['count']:,} training images</span></div>
    </div>
    """)

    html("""
    <div class="sec">
    <h2>How it works</h2>
    <p class="lead">Four steps, all on one page.</p>
    </div>
    <div class="steps">
    <div class="step"><h4>Upload</h4><p>Choose a single CT slice as a PNG or JPG image.</p></div>
    <div class="step"><h4>Prepare</h4><p>The image is resized and its pixel values are scaled to match the training data.</p></div>
    <div class="step"><h4>Classify</h4><p>A deep learning model scores the slice against the three classes.</p></div>
    <div class="step"><h4>Review</h4><p>You see the most likely class and how confident the model is in each one.</p></div>
    </div>
    <div class="notice">
    <strong>Not a medical diagnosis.</strong> This tool is for research and education. Its results must be reviewed by a qualified radiologist or physician and should never be the only basis for a clinical decision.
    </div>
    """)

# ---------------------------------------------------------- analyze page
def analyze():
    st.markdown("## Analyze a CT scan")
    st.caption("Upload one chest CT slice. Images are processed in this session and are not saved.")

    file = st.file_uploader("CT slice (PNG or JPG)", type=["png", "jpg", "jpeg"])
    if file is None:
        html('<div class="hint">Choose an image to begin. A slice through the middle of the chest gives the most reliable result.</div>')
        st.stop()

    raw = file.getvalue()
    img = Image.open(file)
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.image(img, caption=file.name, use_container_width=True)
        run = st.button("Analyze scan", type="primary")

    with right:
        if not run:
            st.write("Press **Analyze scan** to see the result.")
            st.stop()

        with st.spinner("Analyzing…"):
            probs, demo = predict(img, raw)

        top = max(probs, key=probs.get)
        info, conf = CLASSES[top], probs[top]

        if demo:
            html('<div class="demo"><b>Demo mode.</b> No model is connected yet, so these numbers are placeholders, not real predictions.</div>')

        html(f"""
        <div class="result {info['tone']}">
        <div class="tag">{info['title']} · {top}</div>
        <h2>{info['verdict']}</h2>
        <p>{info['detail']}</p>
        <div class="conf">{conf:.1%}<small>model confidence</small></div>
        </div>
        """)

        if conf < LOW_CONFIDENCE:
            html('<div class="warn"><b>Low confidence.</b> The model is not sure about this slice. Have a clinician review the image.</div>')

        rows = ""
        for k, p in sorted(probs.items(), key=lambda kv: -kv[1]):
            rows += f"""<div class="bar-row"><span>{CLASSES[k]['title']} ({k})</span>
            <div class="track"><div class="fill {CLASSES[k]['tone']}" style="width:{p*100:.1f}%"></div></div>
            <span class="pct">{p:.1%}</span></div>"""
        html(f'<div class="bars">{rows}</div>')
        st.caption("For research and education only. Not a medical diagnosis.")

# -------------------------------------------------------------- navigation
if "dark" not in st.session_state:
    try:  # start in the visitor's system theme if Streamlit can tell us
        st.session_state.dark = st.context.theme.type == "dark"
    except Exception:
        st.session_state.dark = False

brand_col, toggle_col = st.columns([5, 1], vertical_alignment="center")
with toggle_col:
    st.toggle("Dark mode", key="dark")
with brand_col:
    html(f"""
    <div class="brand">
    <div class="brand-name">{LOGO} LungLens CT</div>
    <div class="brand-note">AI-assisted chest CT screening · for research use</div>
    </div>
    """)
if st.session_state.dark:
    html('<div class="dark-flag"></div>')
html('<div class="rule"></div>')

home_page = st.Page(home, title="Home", icon=":material/home:", url_path="home", default=True)
analyze_page = st.Page(analyze, title="Analyze a scan", icon=":material/biotech:", url_path="analyze")
st.navigation([home_page, analyze_page], position="top").run()




