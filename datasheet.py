import os
import re
import time
import requests
import fitz  # PyMuPDF
from duckduckgo_search import DDGS

FALLBACK_FOLDER = "sample_datasheets"

# 🔍 Dynamic keyword patterns (partial matches)
KEYWORD_PATTERNS = [
    "MARK",
    "ORDER",
    "PART",
    "DEVICE",
    "IDENT",
    "TOP"
]

# 🚫 Pages to ignore — only checked in first 20% of page text now
IGNORE_WORDS = [
    "REVISION",
    "HISTORY",
    "CHANGE",
    "TABLE OF CONTENTS"
]


# 🔎 Step 1 – Search datasheet links
def search_datasheet(ic_name):
    links = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(f"{ic_name} datasheet pdf", max_results=5)
            for r in results:
                url = r.get("href", "")
                if ".pdf" in url.lower():
                    links.append(url)
    except Exception as e:
        print("Search error:", e)
    return links


# ⬇️ Step 2 – Download PDF
# FIX 1: Timestamp in filename prevents collision between simultaneous runs
# FIX 2: Validate Content-Type to reject HTML pages that contain ".pdf" in URL
def download_pdf(url, ic_name):
    filename = f"{ic_name}_{int(time.time())}.pdf"
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("application/pdf"):
                print(f"Skipping non-PDF response from {url} (Content-Type: {content_type})")
                return None
            with open(filename, "wb") as f:
                f.write(response.content)
            return filename
    except Exception as e:
        print("Download error:", e)
    return None


# 🧠 Smart page scoring
# FIX 3: IGNORE_WORDS only checked in first 20% of page — prevents discarding
#         pages that contain both a TOC reference AND real marking data
def score_page(text, ic_name):
    text_upper = text.upper()
    first_section = text_upper[:len(text_upper) // 5]
    if any(word in first_section for word in IGNORE_WORDS):
        return 0

    score = 0
    if ic_name.upper() in text_upper:
        score += 5
    for pattern in KEYWORD_PATTERNS:
        if pattern in text_upper:
            score += 1
    return score


# 📄 Extract only relevant lines
# FIX 4: Also keeps lines matching KEYWORD_PATTERNS, not just IC name
#         Catches cases like "Top Mark: F103C8T6" where full IC name isn't repeated
def extract_relevant_lines(text, ic_name):
    lines = text.split("\n")
    ic_pattern = re.compile(ic_name, re.IGNORECASE)
    relevant = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if ic_pattern.search(line_stripped):
            relevant.append(line_stripped)
        elif any(kw in line_stripped.upper() for kw in KEYWORD_PATTERNS):
            relevant.append(line_stripped)
    return "\n".join(relevant)


# 📄 Step 3 – Smart marking extractor
# FIX 5: Cleans up downloaded temp file after extraction
def extract_marking_section(pdf_path, ic_name, is_temp=False):
    try:
        doc = fitz.open(pdf_path)
        best_score = 0
        best_text = ""
        for page in doc:
            text = page.get_text()
            score = score_page(text, ic_name)
            if score > best_score:
                best_score = score
                best_text = text
        doc.close()
        result = extract_relevant_lines(best_text, ic_name) if best_text else ""
        return result
    except Exception as e:
        print("PDF read error:", e)
        return ""
    finally:
        # FIX 5: Delete temp downloaded files after reading — keeps project root clean
        if is_temp and os.path.exists(pdf_path):
            os.remove(pdf_path)


# 📂 Step 4 – Load fallback PDFs
def load_fallback_pdfs():
    pdfs = []
    if os.path.exists(FALLBACK_FOLDER):
        for file in os.listdir(FALLBACK_FOLDER):
            if file.endswith(".pdf"):
                pdfs.append(os.path.join(FALLBACK_FOLDER, file))
    return pdfs


# 🚀 Main function – Datasheet Agent
# FIX 6: Returns "" instead of "Marking section not found."
#         Empty string triggers guard clause in verify.py correctly
def get_marking_from_datasheet(ic_name):
    # 1️⃣ Try online search
    links = search_datasheet(ic_name)
    for link in links:
        pdf_file = download_pdf(link, ic_name)
        if pdf_file:
            # is_temp=True so the downloaded file is deleted after reading
            text = extract_marking_section(pdf_file, ic_name, is_temp=True)
            if text:
                return text

    # 2️⃣ Use fallback PDFs
    fallback_pdfs = load_fallback_pdfs()
    for pdf in fallback_pdfs:
        # is_temp=False — don't delete the permanent fallback files
        text = extract_marking_section(pdf, ic_name, is_temp=False)
        if text:
            return text

    # FIX 6: Empty string instead of message string
    return ""