import re

# Supported ranges
UNICODE_RANGES = {
    "Devanagari": (0x0900, 0x097F), # Hindi, Marathi, Nepali, Bhojpuri
    "Bengali": (0x0980, 0x09FF),    # Bengali, Assamese
    "Gurmukhi": (0x0A00, 0x0A7F),   # Punjabi
    "Gujarati": (0x0A80, 0x0AFF),   # Gujarati
    "Odia": (0x0B00, 0x0B7F),       # Odia
    "Tamil": (0x0B80, 0x0BFF),      # Tamil
    "Telugu": (0x0C00, 0x0C7F),     # Telugu
    "Kannada": (0x0C80, 0x0CFF),    # Kannada
    "Malayalam": (0x0D00, 0x0D7F),  # Malayalam
    "Latin": (0x0000, 0x007F),      # English (also transliterated Indic)
    "Arabic": (0x0600, 0x06FF),     # Kashmiri (Perso-Arabic / Urdu script)
}

SCRIPT_TO_LANGUAGES = {
    "Devanagari": ["Hindi", "Marathi", "Nepali", "Bhojpuri"],  # Kashmiri removed — uses Arabic script
    "Bengali": ["Bengali", "Assamese"],
    "Gurmukhi": ["Punjabi"],
    "Gujarati": ["Gujarati"],
    "Odia": ["Odia"],
    "Tamil": ["Tamil"],
    "Telugu": ["Telugu"],
    "Kannada": ["Kannada"],
    "Malayalam": ["Malayalam"],
    "Latin": ["English"],
    "Arabic": ["Kashmiri"],
    "Unknown": ["Hindi", "Marathi", "Bengali", "Assamese", "Odia", "Punjabi", 
                "Gujarati", "Tamil", "Telugu", "Kannada", "Malayalam", 
                "Nepali", "Bhojpuri", "Kashmiri", "English"]
}

def detect_script(text: str) -> str:
    """Detect the predominant script in the input text."""
    if not text:
        return "Unknown"
    
    script_counts = {script: 0 for script in UNICODE_RANGES.keys()}
    clean_text = re.sub(r'\s+', '', text)
    
    if not clean_text:
        return "Unknown"

    for char in clean_text:
        code_point = ord(char)
        for script, (start, end) in UNICODE_RANGES.items():
            if start <= code_point <= end:
                script_counts[script] += 1
                break
                
    detected = max(script_counts.items(), key=lambda x: x[1])
    
    if detected[1] == 0:
        return "Unknown"
        
    return detected[0]

def get_candidate_languages(script: str) -> list:
    """Return candidate languages for a given script."""
    return SCRIPT_TO_LANGUAGES.get(script, SCRIPT_TO_LANGUAGES["Unknown"])
