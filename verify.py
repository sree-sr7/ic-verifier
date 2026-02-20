import os
import json
from groq import Groq
from functools import lru_cache
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """System Role:
You are an automated Quality Assurance (QA) validation engine for high-volume Integrated Circuit (IC) manufacturing. Your sole purpose is to compare text extracted from a physical chip via OCR against the official OEM datasheet specifications.

PRIME DIRECTIVE: ZERO HALLUCINATION & STRICT VALIDATION
1. Never guess, assume, or fabricate missing characters.
2. If either the OCR text or OEM spec is missing, incomplete, or entirely illegible, default to UNVERIFIABLE.
3. Output must be strictly machine-readable JSON only. No conversational filler, no markdown.

Internal Logic Engine:
1. Exact Match Check: Are OCR string and OEM spec perfectly identical? If yes -> GENUINE (Confidence: 95-100%).
2. OCR Error Compensation: If there is a 1-character difference, check if it is a known optical confusion (0 vs O, 1 vs I, 8 vs B, 5 vs S). If purely optical error -> GENUINE (Confidence: 75-90%).
3. Mismatch Detection: If characters are structurally different (wrong manufacturer prefix, missing alphanumeric blocks) -> FAKE (Confidence: 90-100%).
4. Unverifiable Edge Case: If OCR text is noise or OEM spec is empty/Not Found -> UNVERIFIABLE (Confidence: 0-40%).

Output Protocol:
Return ONLY this exact JSON schema, nothing else:
{
  "result": "GENUINE" | "FAKE" | "UNVERIFIABLE",
  "confidence": <integer 0-100>,
  "reasoning": "<1-2 sentence technical explanation>"
}"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_groq_api(user_prompt: str) -> str | None:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"DEBUG ERROR: {type(e).__name__}: {e}")
        return None


def _parse_response(raw: str) -> dict | None:
    """
    Attempts to parse the model's raw text output into a valid verdict dict.
    Returns None if parsing fails or required keys are missing.
    """
    try:
        # The model sometimes wraps JSON in markdown fences like ```json ... ```
        # Strip those out before parsing
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]          # grab content between fences
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]                  # strip the "json" language tag

        verdict = json.loads(cleaned.strip())

        # Validate that all required keys are present and types are correct
        if (
            verdict.get("result") in ("GENUINE", "FAKE", "UNVERIFIABLE")
            and isinstance(verdict.get("confidence"), int)
            and isinstance(verdict.get("reasoning"), str)
        ):
            return verdict

        return None  # Keys exist but values are wrong types/values

    except (json.JSONDecodeError, IndexError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Public API — this is what other modules call
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def verify_ic(scanned_text: str, oem_spec_text: str, ic_part_number: str) -> dict:
    """
    Verifies an IC chip marking against the OEM datasheet specification.

    Caching: @lru_cache means identical inputs return instantly without
    burning an API call. Protects against Groq free tier rate limits (30 RPM)
    during repeated demo runs or testing.

    Args:
        scanned_text:    OCR-extracted text from the physical chip image.
        oem_spec_text:   Expected marking text pulled from the OEM datasheet PDF.
        ic_part_number:  The part number being verified (for context in the prompt).

    Returns:
        A verdict dict with keys: result, confidence, reasoning.
    """

    # ------------------------------------------------------------------
    # Guard clause: no point calling the API if OEM spec is missing
    # ------------------------------------------------------------------
    if not oem_spec_text or not oem_spec_text.strip():
        return {
            "result": "UNVERIFIABLE",
            "confidence": 0,
            "reasoning": "OEM spec text is missing or empty. Cannot perform verification.",
        }

    # ------------------------------------------------------------------
    # Build the user prompt using the required template
    # ------------------------------------------------------------------
    user_prompt = f"""Evaluate the following IC component marking:
- Extracted OCR Text from chip: {scanned_text}
- Expected OEM Specification from datasheet: {oem_spec_text}
- IC Part Number being verified: {ic_part_number}"""

    # ------------------------------------------------------------------
    # First API attempt
    # ------------------------------------------------------------------
    raw_response = _call_groq_api(user_prompt)

    if raw_response is None:
        # Network error or SDK exception — API call failed entirely
        return {
            "result": "UNVERIFIABLE",
            "confidence": 0,
            "reasoning": "API call failed. Unable to reach Groq service.",
        }

    verdict = _parse_response(raw_response)

    # ------------------------------------------------------------------
    # Retry once if the response wasn't valid JSON
    # ------------------------------------------------------------------
    if verdict is None:
        raw_response = _call_groq_api(user_prompt)

        if raw_response is None:
            return {
                "result": "UNVERIFIABLE",
                "confidence": 0,
                "reasoning": "API call failed on retry. Unable to reach Groq service.",
            }

        verdict = _parse_response(raw_response)

    # ------------------------------------------------------------------
    # If still unparseable after retry, give up gracefully
    # ------------------------------------------------------------------
    if verdict is None:
        return {
            "result": "UNVERIFIABLE",
            "confidence": 0,
            "reasoning": "Model returned an unreadable response after two attempts.",
        }

    return verdict


# ---------------------------------------------------------------------------
# Quick manual test — run `python verify.py` directly to check your setup
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = verify_ic(
        scanned_text="STM32F103C8T6",
        oem_spec_text="STM32F103C8T6",
        ic_part_number="STM32F103C8T6",
    )
    print(json.dumps(result, indent=2))