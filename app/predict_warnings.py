"""Rule-based warning flags derived from job text (complements model scores)."""

from __future__ import annotations

import re
from typing import Final

# (warning_code, regex) — codes are stable for clients; match case-insensitively.
_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    (
        "upfront_payment",
        re.compile(
            r"\b(?:processing fee|registration fee|advance payment|pay (?:first|upfront|in advance)|"
            r"pay\s+\$?\s*\d+|send (?:money|payment|funds)|wire transfer|western union|moneygram|"
            r"training deposit|security deposit|refundable fee)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "off_platform_contact",
        re.compile(
            r"\b(?:telegram|whatsapp|signal(?:\.me)?|viber|wechat|line (?:id|app)|"
            r"contact (?:only|us) (?:on|via))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "high_pressure",
        re.compile(
            r"\b(?:urgent(?:ly)?|limited (?:time|slots|positions)|act now|apply immediately|"
            r"today only|first come|respond immediately)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "guaranteed_income",
        re.compile(
            r"\b(?:guaranteed (?:income|salary|pay|earnings|placement)|"
            r"no interview (?:needed|required)|earn (?:fast |easy )?money)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "crypto_or_gift_card",
        re.compile(
            r"\b(?:bitcoin|btc|ethereum|eth\b|usdt|cryptocurrency|crypto wallet|"
            r"gift card|itunes|google play card)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sensitive_info_request",
        re.compile(
            r"\b(?:ssn|social security|bank account|routing number|pin code|"
            r"upload (?:your )?(?:id|passport|license))\b",
            re.IGNORECASE,
        ),
    ),
]


def compute_warnings(text: str) -> list[str]:
    """Return sorted unique warning codes for the given combined job text."""
    if not text or not str(text).strip():
        return []
    s = str(text)
    found: set[str] = set()
    for code, pat in _PATTERNS:
        if pat.search(s):
            found.add(code)
    return sorted(found)
