import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

model = YOLO(MODEL_PATH)

st.title("🧵 AI Fabric Quality Inspection System")
st.write("Upload a fabric image to detect defects.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name, "JPEG")
        results = model(tmp.name)

    annotated_image = results[0].plot()

    st.subheader("🔍 Detection Result")
    st.image(annotated_image, use_container_width=True)

    if len(results[0].boxes) > 0:
        st.error(f"❌ Status: NG ({len(results[0].boxes)} defects found)")
        for box in results[0].boxes:
            st.write(
                f"- {model.names[int(box.cls[0])]} "
                f"(Conf: {float(box.conf[0]):.2f})"
            )
    else:
        st.success("✅ Status: OK")
