import cv2
import numpy as np
import easyocr
import os

# --- 1. SETUP ---
# Strict allowlist prevents hallucinated symbols
ALLOWED_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
reader = easyocr.Reader(['en'], gpu=False)

def preprocess_for_ic(img_path):
    """The Colab-Tested Winning Pipeline for Laser Dots"""
    img = cv2.imread(img_path)
    if img is None: return None

    # 1. Upscale
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Median Blur (Kills salt-and-pepper noise)
    gray = cv2.medianBlur(gray, 3)

    # 3. Bilateral Filter
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)

    # 4. Auto-Inversion
    h, w = smoothed.shape
    center_roi = smoothed[h//4:3*h//4, w//4:3*w//4]
    if np.mean(center_roi) < 127:
        smoothed = cv2.bitwise_not(smoothed)

    # 5. Adaptive Threshold (Tuned to 45, 15)
    thresh = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 45, 15
    )

    # 6. Erosion (Connects the laser dots!)
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.erode(thresh, kernel, iterations=1)

    # Save to local debug folder
    os.makedirs("debug", exist_ok=True)
    cv2.imwrite(os.path.join("debug", "latest_scan.png"), processed)
    
    return processed

def get_ocr_text(image_path):
    """
    Main function called by app.py.
    Returns: (text: str, confidence: float)
    """
    processed_img = preprocess_for_ic(image_path)
    if processed_img is None: 
        return "Error: Image not found", 0.0

    # Run OCR with constraints
    results = reader.readtext(processed_img, allowlist=ALLOWED_CHARS, min_size=10)

    texts = []
    confidences = []

    for (bbox, text, prob) in results:
        text = text.strip().upper()
        # Keep only strong matches
        if len(text) >= 3 and prob > 0.15:
            texts.append(text)
            confidences.append(prob)

    if not texts:
        return "No text detected", 0.0

    final_text = " ".join(texts)
    avg_conf = (sum(confidences) / len(confidences)) * 100

    return final_text, round(avg_conf, 2)