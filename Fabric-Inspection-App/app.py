```python
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Fabric Quality Inspection",
    page_icon="🧵",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp{
    background-color:#f5f7fa;
}

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1e293b;
}

.sub-title{
    text-align:center;
    color:#64748b;
    font-size:18px;
    margin-bottom:25px;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:10px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="main-title">
🧵 AI Fabric Quality Inspection System
</div>

<div class="sub-title">
Automated Fabric Defect Detection using YOLOv8
</div>
""", unsafe_allow_html=True)

# =====================================================
# DASHBOARD
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "YOLOv8")

with col2:
    st.metric("Inspection", "Active")

with col3:
    st.metric("System", "Online")

st.markdown("---")

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "📤 Upload Fabric Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# PREDICTION
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name, "JPEG")

        with st.spinner("🔍 Inspecting Fabric..."):
            results = model(tmp.name)

    annotated_image = results[0].plot()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Fabric Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🎯 Detection Result")
        st.image(annotated_image, use_container_width=True)

    st.markdown("---")

    boxes = results[0].boxes

    # =================================================
    # STATUS
    # =================================================

    if len(boxes) > 0:

        st.error(
            f"🚨 QUALITY FAILURE DETECTED | {len(boxes)} Defect(s) Found"
        )

        st.subheader("📋 Defect Details")

        defect_count = 1

        for box in boxes:

            defect_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0]) * 100

            st.write(
                f"**{defect_count}. {defect_name}** | Confidence: {confidence:.1f}%"
            )

            defect_count += 1

    else:

        st.success(
            "✅ QUALITY APPROVED - No Defects Found"
        )

        st.balloons()

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Developed using Streamlit + YOLOv8 for Automated Fabric Defect Detection"
)
```
