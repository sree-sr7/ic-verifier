import os
import json
from datetime import datetime
from groq import Groq
from functools import lru_cache
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.1-8b-instant"
CACHE_FILE = "verified_cache.json"

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
# Human-Verified Cache — feedback loop so the system learns over time
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    """Loads the human-verified cache from disk. Returns empty dict if not found."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_to_cache(ic_part_number: str, result: str, oem_spec: str, notes: str = "") -> None:
    """
    Called by Person 4's Streamlit UI when a human QA engineer manually
    verifies an UNVERIFIABLE result.

    Args:
        ic_part_number: The IC part number e.g. "STM32F103C8T6"
        result:         Human decision — "GENUINE" or "FAKE"
        oem_spec:       The OEM spec text used during verification
        notes:          Optional notes from the QA engineer
    
    Usage in Streamlit (Person 4):
        from verify import save_to_cache
        save_to_cache("STM32F103C8T6", "GENUINE", oem_spec_text, "Verified under magnifier")
    """
    if result not in ("GENUINE", "FAKE"):
        print(f"Invalid result '{result}' — must be GENUINE or FAKE")
        return

    cache = load_cache()
    cache[ic_part_number] = {
        "result": result,
        "verified_by": "human",
        "oem_spec": oem_spec,
        "timestamp": datetime.now().isoformat(),
        "notes": notes
    }
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"✅ Saved to cache: {ic_part_number} → {result}")
    except IOError as e:
        print(f"Cache write error: {e}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_groq_api(user_prompt: str) -> str | None:
    """
    Sends the prompt to Groq and returns raw response string.
    Returns None on any failure.
    """
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
    Parses the model's raw text output into a valid verdict dict.
    Returns None if parsing fails or required keys are missing.
    """
    try:
        # Strip markdown fences if model wraps response in ```json ... ```
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]

        verdict = json.loads(cleaned.strip())

        # Validate all required keys and types
        if (
            verdict.get("result") in ("GENUINE", "FAKE", "UNVERIFIABLE")
            and isinstance(verdict.get("confidence"), int)
            and isinstance(verdict.get("reasoning"), str)
        ):
            return verdict

        return None

    except (json.JSONDecodeError, IndexError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Public API — this is what other modules call
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def verify_ic(scanned_text: str, oem_spec_text: str, ic_part_number: str) -> dict:
    """
    Verifies an IC chip marking against the OEM datasheet specification.

    Priority order:
      1. Human-verified cache — instant result, no API call, 99% confidence
      2. Guard clause — empty OEM spec returns UNVERIFIABLE immediately
      3. Groq LLM — full AI comparison with retry logic

    Caching: @lru_cache prevents duplicate API calls for identical inputs
    during the same session, protecting against Groq free tier rate limits.

    Args:
        scanned_text:    OCR-extracted text from the physical chip image.
        oem_spec_text:   Expected marking text pulled from the OEM datasheet PDF.
        ic_part_number:  The part number being verified (for context in the prompt).

    Returns:
        A verdict dict with keys: result, confidence, reasoning.
    """

    # ------------------------------------------------------------------
    # 1. Check human-verified cache FIRST
    #    If a human QA engineer has already reviewed this IC, trust that
    #    decision instantly — no need to call the API again
    # ------------------------------------------------------------------
    cache = load_cache()
    if ic_part_number in cache:
        cached = cache[ic_part_number]
        print(f"📦 Cache hit: {ic_part_number} — returning human-verified result")
        return {
            "result": cached["result"],
            "confidence": 99,
            "reasoning": f"Previously verified by human QA on {cached['timestamp']}. {cached.get('notes', '')}".strip()
        }

    # ------------------------------------------------------------------
    # 2. Guard clause — no point calling the API if OEM spec is missing
    # ------------------------------------------------------------------
    if not oem_spec_text or not oem_spec_text.strip():
        return {
            "result": "UNVERIFIABLE",
            "confidence": 0,
            "reasoning": "OEM spec text is missing or empty. Cannot perform verification.",
        }

    # ------------------------------------------------------------------
    # 3. Build the user prompt
    # ------------------------------------------------------------------
    user_prompt = f"""Evaluate the following IC component marking:
- Extracted OCR Text from chip: {scanned_text}
- Expected OEM Specification from datasheet: {oem_spec_text}
- IC Part Number being verified: {ic_part_number}"""

    # ------------------------------------------------------------------
    # 4. First API attempt
    # ------------------------------------------------------------------
    raw_response = _call_groq_api(user_prompt)

    if raw_response is None:
        return {
            "result": "UNVERIFIABLE",
            "confidence": 0,
            "reasoning": "API call failed. Unable to reach Groq service.",
        }

    verdict = _parse_response(raw_response)

    # ------------------------------------------------------------------
    # 5. Retry once if response wasn't valid JSON
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
    # 6. Give up gracefully if still unparseable
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
    # Test 1: Normal LLM verification
    print("=== Test 1: LLM Verification ===")
    result = verify_ic(
        scanned_text="STM32F103C8T6",
        oem_spec_text="STM32F103C8T6",
        ic_part_number="STM32F103C8T6",
    )
    print(json.dumps(result, indent=2))

    # Test 2: Save a human verification to cache
    print("\n=== Test 2: Saving to Cache ===")
    save_to_cache(
        ic_part_number="NE555P",
        result="GENUINE",
        oem_spec="NE555P",
        notes="Verified under magnifier by QA engineer"
    )

    # Test 3: Cache hit — should return instantly without API call
    print("\n=== Test 3: Cache Hit ===")
    result = verify_ic(
        scanned_text="NE555P",
        oem_spec_text="NE555P",
        ic_part_number="NE555P",
    )
    print(json.dumps(result, indent=2))