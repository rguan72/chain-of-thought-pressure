"""Parser for extracting CoT and final answer from model responses."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedResponse:
    """Parsed model response with CoT and final answer."""

    cot_text: str
    final_text: str
    format_ok: bool
    raw_response: str

    @property
    def has_cot(self) -> bool:
        """Whether CoT was found."""
        return bool(self.cot_text)

    @property
    def has_final(self) -> bool:
        """Whether final answer was found."""
        return bool(self.final_text)


def parse_cot_response(response: str) -> ParsedResponse:
    """Parse a model response to extract CoT and final answer.

    Expected format:
    <think>
    [reasoning]
    </think>
    <final>
    [answer]
    </final>

    Args:
        response: Raw model response string.

    Returns:
        ParsedResponse with extracted fields.
    """
    cot_text = ""
    final_text = ""

    # Extract content between <think> and </think> tags
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, response, re.DOTALL | re.IGNORECASE)
    if think_match:
        cot_text = think_match.group(1).strip()

    # Extract content between <final> and </final> tags
    final_pattern = r"<final>(.*?)</final>"
    final_match = re.search(final_pattern, response, re.DOTALL | re.IGNORECASE)
    if final_match:
        final_text = final_match.group(1).strip()

    # Fallback: if no <final> tag, try to get content after </think>
    if not final_text and think_match:
        remaining = response[think_match.end() :].strip()
        # Remove any tags and get the text
        remaining = re.sub(r"<[^>]+>", "", remaining).strip()
        if remaining:
            final_text = remaining

    # Further fallback: if nothing parsed, use the whole response as final
    if not final_text and not cot_text:
        # Try to use the last substantive part of the response
        cleaned = re.sub(r"<[^>]+>", "", response).strip()
        if cleaned:
            final_text = cleaned

    # Determine if format is OK (both tags present and valid)
    format_ok = bool(think_match and final_match and cot_text and final_text)

    return ParsedResponse(
        cot_text=cot_text, final_text=final_text, format_ok=format_ok, raw_response=response
    )


def extract_cot_only(response: str) -> Optional[str]:
    """Extract only the CoT portion from a response.

    Args:
        response: Raw model response.

    Returns:
        CoT text if found, None otherwise.
    """
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_final_only(response: str) -> Optional[str]:
    """Extract only the final answer from a response.

    Args:
        response: Raw model response.

    Returns:
        Final text if found, None otherwise.
    """
    pattern = r"<final>(.*?)</final>"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
