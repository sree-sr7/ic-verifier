import os
import re
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

# 🚫 Pages to ignore
IGNORE_WORDS = [
    "REVISION",
    "HISTORY",
    "CHANGE",
    "TABLE OF CONTENTS"
]


# 🔎 Step 1 – Search datasheet links
def search_datasheet(ic_name):
    links = []

    with DDGS() as ddgs:
        results = ddgs.text(f"{ic_name} datasheet pdf", max_results=5)

        for r in results:
            url = r.get("href", "")
            if ".pdf" in url.lower():
                links.append(url)

    return links


# ⬇️ Step 2 – Download PDF
def download_pdf(url, filename="datasheet.pdf"):
    try:
        response = requests.get(url, timeout=20)

        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            return filename

    except Exception as e:
        print("Download error:", e)

    return None


# 🧠 Smart page scoring
def score_page(text, ic_name):
    text_upper = text.upper()

    if any(word in text_upper for word in IGNORE_WORDS):
        return 0

    score = 0

    # IC name presence
    if ic_name.upper() in text_upper:
        score += 5

    # Keyword pattern matches
    for pattern in KEYWORD_PATTERNS:
        if pattern in text_upper:
            score += 1

    return score


# 📄 Extract only relevant lines
def extract_relevant_lines(text, ic_name):
    lines = text.split("\n")
    ic_pattern = re.compile(ic_name, re.IGNORECASE)

    relevant = []

    for line in lines:
        if ic_pattern.search(line):
            relevant.append(line.strip())

    return "\n".join(relevant)


# 📄 Step 3 – Smart marking extractor
def extract_marking_section(pdf_path, ic_name):
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

        if best_text:
            return extract_relevant_lines(best_text, ic_name)

        return ""

    except Exception as e:
        print("PDF read error:", e)
        return ""


# 📂 Step 4 – Load fallback PDFs
def load_fallback_pdfs():
    pdfs = []

    if os.path.exists(FALLBACK_FOLDER):
        for file in os.listdir(FALLBACK_FOLDER):
            if file.endswith(".pdf"):
                pdfs.append(os.path.join(FALLBACK_FOLDER, file))

    return pdfs


# 🚀 Main function – Datasheet Agent
def get_marking_from_datasheet(ic_name):
    # 1️⃣ Try online search
    links = search_datasheet(ic_name)

    for link in links:
        pdf_file = download_pdf(link, f"{ic_name}.pdf")

        if pdf_file:
            text = extract_marking_section(pdf_file, ic_name)

            if text:
                return text

    # 2️⃣ Use fallback PDFs
    fallback_pdfs = load_fallback_pdfs()

    for pdf in fallback_pdfs:
        text = extract_marking_section(pdf, ic_name)

        if text:
            return text

    return "Marking section not found."