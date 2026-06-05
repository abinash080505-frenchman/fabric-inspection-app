import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load the model you just trained
model = YOLO("Fabric-Inspection-App/best.pt")

st.title("🧵 AI Fabric Quality Inspection System")
st.write("Upload a fabric image to detect defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Temporary save for YOLO
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    image = image.convert("RGB")
    image.save(tmp.name, format="JPEG")
    results = model(tmp.name)

    # Show Result
    annotated_image = results[0].plot()
    st.subheader("🔍 Detection Result")
    st.image(annotated_image, use_container_width=True)

    if len(results[0].boxes) > 0:
        st.error(f"❌ Status: NG ({len(results[0].boxes)} defects found)")
        for box in results[0].boxes:
            st.write(f"- {model.names[int(box.cls[0])]} (Conf: {float(box.conf[0]):.2f})")
    else:
        st.success("✅ Status: OK")
