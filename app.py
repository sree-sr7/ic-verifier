import streamlit as st
import cv2
import numpy as np
from PIL import Image
from verify import verify_ic
from ocr import get_ocr_text
from datasheet import get_marking_from_datasheet

# ---------------- UI ----------------
st.set_page_config(page_title="IC Verifier", layout="wide")
st.title("IC Fake Detection System")
st.markdown("### AI-Powered IC Authenticity Verification")

page = st.sidebar.radio("Navigation", ["Verifier", "Architecture"])
demo_mode = st.sidebar.toggle("Enable Fake Demo Mode")
st.sidebar.title("About")
st.sidebar.write("IC Fake Detection using AI")
st.sidebar.write("Hackathon Demo Project")
st.sidebar.write("Pipeline: OCR → Datasheet → LLM")

# ---------------- Architecture Page ----------------
if page == "Architecture":
    st.header("System Architecture")
    st.markdown("""
**Pipeline Flow**

Image → OpenCV Preprocessing → OCR → Datasheet Agent → LLM Verification → Verdict
""")
    st.info("Modular AI Agent pipeline for IC authenticity verification.")

# ---------------- Verifier Page ----------------
if page == "Verifier":
    uploaded_file = st.file_uploader("Upload IC Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        # Save uploaded file temporarily for ocr.py (needs file path)
        temp_path = "temp_ic.png"
        image.save(temp_path)

        # Show original image
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

        # OCR
        with st.spinner("Running OCR..."):
            text, ocr_confidence = get_ocr_text(temp_path)

        st.subheader("Extracted Marking")
        st.success(text if text else "No text extracted")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("OCR Confidence", f"{ocr_confidence}%")
        with col2:
            st.success("✔ OCR completed")

        # Clean marking
        clean_marking = text.split()[0] if text else ""

        # Fake demo mode
        if demo_mode and clean_marking:
            clean_marking = clean_marking + "X"
            st.warning("⚠ Demo Mode: Simulating Fake IC by altering marking")

        # Datasheet lookup
        with st.spinner("Searching for datasheet..."):
            oem_spec_text = get_marking_from_datasheet(clean_marking)

        if oem_spec_text:
            st.success("✔ Datasheet retrieved")
            with st.expander("View OEM Spec Text"):
                st.text(oem_spec_text)
        else:
            st.warning("⚠ Datasheet not found - result will be UNVERIFIABLE")

        # AI verification
        with st.spinner("AI verification running..."):
            verdict_data = verify_ic(
                scanned_text=text,
                oem_spec_text=oem_spec_text,
                ic_part_number=clean_marking,
            )

        # Verdict display
        st.markdown("---")
        st.markdown("## Verification Result")

        result = verdict_data["result"]
        confidence_value = int(verdict_data["confidence"])

        if result == "GENUINE":
            st.success("✅ GENUINE")
        elif result == "FAKE":
            st.error("🚨 FAKE DETECTED")
        else:
            st.warning("⚠ UNVERIFIABLE")

        st.progress(confidence_value)
        st.metric("Confidence Score", f"{confidence_value}%")
        st.markdown(f"**Reasoning:** {verdict_data['reasoning']}")
