"""
romanized.py
------------
Handles Romanized (Latin-script) Indic text:
  1. is_latin_dominant(text) — detects if text is mostly Latin
  2. get_transliterations(text) — returns a dict of {script_name: transliterated_text}
     using the indic-transliteration library (HK scheme → target scripts).
"""

import re

# ---------------------------------------------------------------------------
# Step 1 – Detection
# ---------------------------------------------------------------------------

def is_latin_dominant(text: str) -> bool:
    """Return True if >60% of the characters are a-z / A-Z (Latin alphabet)."""
    if not text:
        return False
    latin_chars = sum('a' <= c.lower() <= 'z' for c in text)
    return latin_chars / max(len(text), 1) > 0.6


# ---------------------------------------------------------------------------
# Step 2 – Transliteration
# ---------------------------------------------------------------------------

# Map from indic-transliteration scheme names to human-readable labels
# (the library uses these string constants for target scripts)
_TARGET_SCHEMES = {
    "Devanagari": "DEVANAGARI",
    "Bengali":    "BENGALI",
    "Odia":       "ORIYA",      # library constant is ORIYA for Odia script
    "Gurmukhi":   "GURMUKHI",
    "Gujarati":   "GUJARATI",
    "Tamil":      "TAMIL",
    "Telugu":     "TELUGU",
    "Kannada":    "KANNADA",
    "Malayalam":  "MALAYALAM",
    # Arabic/Urdu script: not supported by indic-transliteration,
    # the except block in get_transliterations will skip it safely.
}


def get_transliterations(text: str) -> dict:
    """
    Transliterate `text` (assumed to be Romanized IAST/HK-like Latin input)
    into each of the supported Indic scripts.

    Returns a dict like:
      {
        "Devanagari": "<devanagari text>",
        "Bengali":    "<bengali text>",
        ...
      }

    Falls back to an empty dict if the library is unavailable.
    """
    results = {}

    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        # We treat the input as ITRANS (a common romanisation scheme).
        # ITRANS is the most permissive scheme for arbitrary Romanized Indic text.
        source_scheme = sanscript.ITRANS

        for label, target_scheme_name in _TARGET_SCHEMES.items():
            try:
                target_scheme = getattr(sanscript, target_scheme_name.upper())
                transliterated = transliterate(text, source_scheme, target_scheme)
                if transliterated and transliterated.strip():
                    results[label] = transliterated.strip()
            except Exception as e:
                print(f"[romanized] Skipping {label}: {e}")

    except ImportError:
        print("[romanized] 'indic-transliteration' not installed. "
              "Run: pip install indic-transliteration")

    return results
