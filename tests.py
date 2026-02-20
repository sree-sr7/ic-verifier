"""
tests.py — Unit tests for verify.py
Run with: python tests.py
"""

import unittest
from unittest.mock import patch, MagicMock
from verify import verify_ic, _parse_response, _call_groq_api

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_groq_response(content: str):
    """
    Fakes a Groq API response object so we never hit the real API during tests.
    Mirrors the structure of: response.choices[0].message.content
    """
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    return mock_response


GENUINE_JSON    = '{"result": "GENUINE", "confidence": 95, "reasoning": "Exact match found."}'
FAKE_JSON       = '{"result": "FAKE", "confidence": 92, "reasoning": "Structural mismatch detected."}'
UNVERIFIABLE_JSON = '{"result": "UNVERIFIABLE", "confidence": 10, "reasoning": "OCR text is noise."}'
MARKDOWN_JSON   = '```json\n{"result": "GENUINE", "confidence": 80, "reasoning": "OCR optical confusion only."}\n```'
BAD_JSON        = "Sorry, I cannot determine this. Please try again."


# ===========================================================================
# 1. Guard Clause Tests — these never call the API
# ===========================================================================

class TestGuardClauses(unittest.TestCase):

    def test_empty_oem_spec_returns_unverifiable(self):
        """Empty string OEM spec should short-circuit before API call."""
        result = verify_ic("STM32F103C8T6", "", "STM32F103C8T6")
        self.assertEqual(result["result"], "UNVERIFIABLE")
        self.assertEqual(result["confidence"], 0)

    def test_none_oem_spec_returns_unverifiable(self):
        """None OEM spec should short-circuit before API call."""
        result = verify_ic("STM32F103C8T6", None, "STM32F103C8T6")
        self.assertEqual(result["result"], "UNVERIFIABLE")
        self.assertEqual(result["confidence"], 0)

    def test_whitespace_only_oem_spec_returns_unverifiable(self):
        """Whitespace-only OEM spec should be treated as empty."""
        result = verify_ic("STM32F103C8T6", "   ", "STM32F103C8T6")
        self.assertEqual(result["result"], "UNVERIFIABLE")
        self.assertEqual(result["confidence"], 0)


# ===========================================================================
# 2. JSON Parser Tests — tests _parse_response() directly
# ===========================================================================

class TestParseResponse(unittest.TestCase):

    def test_parses_clean_genuine_json(self):
        verdict = _parse_response(GENUINE_JSON)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["result"], "GENUINE")
        self.assertEqual(verdict["confidence"], 95)

    def test_parses_clean_fake_json(self):
        verdict = _parse_response(FAKE_JSON)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["result"], "FAKE")

    def test_parses_clean_unverifiable_json(self):
        verdict = _parse_response(UNVERIFIABLE_JSON)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["result"], "UNVERIFIABLE")

    def test_strips_markdown_fences(self):
        """Model sometimes wraps JSON in ```json ... ``` — must be handled."""
        verdict = _parse_response(MARKDOWN_JSON)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["result"], "GENUINE")

    def test_returns_none_for_bad_json(self):
        """Non-JSON response should return None so retry logic kicks in."""
        verdict = _parse_response(BAD_JSON)
        self.assertIsNone(verdict)

    def test_returns_none_for_invalid_result_value(self):
        """result must be exactly GENUINE, FAKE, or UNVERIFIABLE."""
        bad = '{"result": "MAYBE", "confidence": 50, "reasoning": "hmm"}'
        verdict = _parse_response(bad)
        self.assertIsNone(verdict)

    def test_returns_none_for_non_integer_confidence(self):
        """confidence must be an int, not a float or string."""
        bad = '{"result": "GENUINE", "confidence": "95", "reasoning": "ok"}'
        verdict = _parse_response(bad)
        self.assertIsNone(verdict)

    def test_returns_none_for_empty_string(self):
        verdict = _parse_response("")
        self.assertIsNone(verdict)


# ===========================================================================
# 3. API Failure Tests — mocks the Groq client so no real calls are made
# ===========================================================================

class TestAPIFailures(unittest.TestCase):

    @patch("verify.client.chat.completions.create")
    def test_api_network_failure_returns_unverifiable(self, mock_create):
        """Simulates a network crash — should return UNVERIFIABLE gracefully."""
        mock_create.side_effect = Exception("Connection timeout")

        result = verify_ic("STM32F103C8T6", "STM32F103C8T6", "STM32F103C8T6")

        self.assertEqual(result["result"], "UNVERIFIABLE")
        self.assertEqual(result["confidence"], 0)
        self.assertIn("failed", result["reasoning"].lower())

    @patch("verify.client.chat.completions.create")
    def test_bad_json_retries_once_then_unverifiable(self, mock_create):
        """
        Simulates model returning bad JSON twice.
        Should retry once and then return UNVERIFIABLE — not crash.
        """
        mock_create.return_value = make_groq_response(BAD_JSON)

        result = verify_ic("STM32F103C8T6", "STM32F103C8T6", "STM32F103C8T6")

        self.assertEqual(result["result"], "UNVERIFIABLE")
        self.assertEqual(mock_create.call_count, 2)  # called twice = 1 attempt + 1 retry

    @patch("verify.client.chat.completions.create")
    def test_bad_json_first_then_good_json_succeeds(self, mock_create):
        """
        Simulates model returning bad JSON on first try, good JSON on retry.
        Should succeed on the second attempt.
        """
        mock_create.side_effect = [
            make_groq_response(BAD_JSON),      # first call — bad
            make_groq_response(GENUINE_JSON),  # retry — good
        ]

        result = verify_ic("STM32F103C8T6", "STM32F103C8T6", "STM32F103C8T6")

        self.assertEqual(result["result"], "GENUINE")
        self.assertEqual(mock_create.call_count, 2)


# ===========================================================================
# 4. Happy Path Tests — mocks Groq to return expected verdicts
# ===========================================================================

class TestHappyPaths(unittest.TestCase):

    @patch("verify.client.chat.completions.create")
    def test_exact_match_returns_genuine(self, mock_create):
        """Identical OCR and OEM spec should return GENUINE."""
        mock_create.return_value = make_groq_response(GENUINE_JSON)

        result = verify_ic("STM32F103C8T6", "STM32F103C8T6", "STM32F103C8T6")

        self.assertEqual(result["result"], "GENUINE")
        self.assertGreaterEqual(result["confidence"], 75)
        self.assertIsInstance(result["reasoning"], str)

    @patch("verify.client.chat.completions.create")
    def test_mismatch_returns_fake(self, mock_create):
        """Structurally different marking should return FAKE."""
        mock_create.return_value = make_groq_response(FAKE_JSON)

        result = verify_ic("STM32F103C8T5", "STM32F103C8T6", "STM32F103C8T6")

        self.assertEqual(result["result"], "FAKE")
        self.assertGreaterEqual(result["confidence"], 80)

    @patch("verify.client.chat.completions.create")
    def test_markdown_wrapped_response_is_handled(self, mock_create):
        """Model sometimes returns ```json ... ``` — must still parse correctly."""
        mock_create.return_value = make_groq_response(MARKDOWN_JSON)

        result = verify_ic("STM32F1O3C8T6", "STM32F103C8T6", "STM32F103C8T6")

        self.assertEqual(result["result"], "GENUINE")

    @patch("verify.client.chat.completions.create")
    def test_verdict_dict_always_has_required_keys(self, mock_create):
        """No matter what, the returned dict must always have all 3 keys."""
        mock_create.return_value = make_groq_response(GENUINE_JSON)

        result = verify_ic("STM32F103C8T6", "STM32F103C8T6", "STM32F103C8T6")

        self.assertIn("result", result)
        self.assertIn("confidence", result)
        self.assertIn("reasoning", result)


# ===========================================================================
# 5. Demo Test Cases — mirrors your 3 actual hackathon demo scenarios
# ===========================================================================

class TestDemoScenarios(unittest.TestCase):

    @patch("verify.client.chat.completions.create")
    def test_demo_genuine_case(self, mock_create):
        """Demo Case 1: Real IC, real datasheet — should be GENUINE."""
        mock_create.return_value = make_groq_response(GENUINE_JSON)

        result = verify_ic(
            scanned_text="STM32F103C8T6",
            oem_spec_text="STM32F103C8T6",
            ic_part_number="STM32F103C8T6"
        )
        self.assertEqual(result["result"], "GENUINE")

    @patch("verify.client.chat.completions.create")
    def test_demo_fake_case(self, mock_create):
        """Demo Case 2: Altered marking — should be FAKE."""
        mock_create.return_value = make_groq_response(FAKE_JSON)

        result = verify_ic(
            scanned_text="STM32F103C8T5",   # last char altered to simulate counterfeit
            oem_spec_text="STM32F103C8T6",
            ic_part_number="STM32F103C8T6"
        )
        self.assertEqual(result["result"], "FAKE")

    def test_demo_unverifiable_case(self):
        """Demo Case 3: No datasheet found — should be UNVERIFIABLE without API call."""
        result = verify_ic(
            scanned_text="UNKNOWN_IC_XR2206",
            oem_spec_text="",               # Person 2 found nothing
            ic_part_number="XR2206"
        )
        self.assertEqual(result["result"], "UNVERIFIABLE")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)