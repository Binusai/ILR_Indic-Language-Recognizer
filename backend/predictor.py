from .script_detector import detect_script, get_candidate_languages
from .model_loader import global_model_loader
from .phoneme_utils import text_to_phonemes
from .romanized import is_latin_dominant, get_transliterations

ALL_LANGUAGES = [
    "Hindi", "Marathi", "Bengali", "Assamese", "Odia",
    "Punjabi", "Gujarati", "Tamil", "Telugu", "Kannada",
    "Malayalam", "Nepali", "Bhojpuri", "Kashmiri", "English"
]

# Scripts where L2 alone cannot reliably distinguish — always use L3
ALWAYS_USE_L3_SCRIPTS = {"Bengali"}  # Bengali & Assamese share identical script


# ---------------------------------------------------------------------------
# Core single-text pipeline (reused for both normal and Romanized paths)
# ---------------------------------------------------------------------------

def _run_pipeline(text: str) -> dict:
    """
    Runs the 3-layer ILR pipeline on a single piece of text and returns the
    full result dict (layer1_script, layer1_candidates, layer2_top3,
    layer3_top3, final_prediction, final_confidence).
    """
    # --- LAYER 1: Script Detection ---
    detected_script = detect_script(text)
    script_candidates = get_candidate_languages(detected_script)

    # --- LAYER 2: Transformer Model (all 15 for display) ---
    layer2_all_probs = global_model_loader.predict_layer2(text, ALL_LANGUAGES)

    # APPLY HINDI/BHOJPURI RULE
    if "Hindi" in layer2_all_probs and "Bhojpuri" in layer2_all_probs:
        hindi_prob = layer2_all_probs["Hindi"]
        bhoj_prob  = layer2_all_probs["Bhojpuri"]
        if abs(hindi_prob - bhoj_prob) < 0.05:
            shift_amount = hindi_prob * 0.60
            layer2_all_probs["Hindi"]    -= shift_amount
            layer2_all_probs["Bhojpuri"] += shift_amount

    sorted_l2_all = sorted(layer2_all_probs.items(), key=lambda x: x[1], reverse=True)
    layer2_top3_display = [[lang, round(prob, 3)] for lang, prob in sorted_l2_all[:3]]

    # --- LAYER 3: Phoneme Model (runs on L2's top 3 languages) ---
    l2_top3_langs = [lang for lang, _ in sorted_l2_all[:3]]

    phonemes = text_to_phonemes(text)
    layer3_all_probs = global_model_loader.predict_layer3(phonemes, l2_top3_langs)
    sorted_l3_all = sorted(layer3_all_probs.items(), key=lambda x: x[1], reverse=True)
    layer3_top3_display = [[lang, round(prob, 3)] for lang, prob in sorted_l3_all[:3]]

    # === FINAL PREDICTION LOGIC ===

    if len(script_candidates) == 1:
        # Only one candidate → that's the answer
        final_prediction = script_candidates[0]
        l2_conf = layer2_all_probs.get(final_prediction, 0.0)
        l3_conf = layer3_all_probs.get(final_prediction, 0.0)
        final_confidence = round(l2_conf + l3_conf, 3)

    elif detected_script in ALWAYS_USE_L3_SCRIPTS:
        # Bengali/Assamese: always use L3 among L1 candidates
        l3_filtered = {lang: layer3_all_probs.get(lang, 0.0) for lang in script_candidates}
        sorted_l3_filtered = sorted(l3_filtered.items(), key=lambda x: x[1], reverse=True)

        best_l3_lang  = sorted_l3_filtered[0][0]
        best_l3_prob  = sorted_l3_filtered[0][1]
        second_l3_prob = sorted_l3_filtered[1][1] if len(sorted_l3_filtered) > 1 else 0.0

        if best_l3_prob > second_l3_prob:
            final_prediction = best_l3_lang
            l2_conf = layer2_all_probs.get(best_l3_lang, 0.0)
            final_confidence = round(best_l3_prob + l2_conf, 3)
        else:
            l2_filtered = {lang: layer2_all_probs.get(lang, 0.0) for lang in script_candidates}
            sorted_l2_filtered = sorted(l2_filtered.items(), key=lambda x: x[1], reverse=True)
            final_prediction = sorted_l2_filtered[0][0]
            final_confidence = round(sorted_l2_filtered[0][1], 3)

    else:
        # General case (Devanagari, etc.)
        l2_filtered = {lang: layer2_all_probs.get(lang, 0.0) for lang in script_candidates}
        sorted_l2_filtered = sorted(l2_filtered.items(), key=lambda x: x[1], reverse=True)
        l2_best_lang = sorted_l2_filtered[0][0]
        l2_best_prob = sorted_l2_filtered[0][1]

        l3_filtered = {lang: layer3_all_probs.get(lang, 0.0) for lang in script_candidates}
        sorted_l3_filtered = sorted(l3_filtered.items(), key=lambda x: x[1], reverse=True)
        l3_best_lang = sorted_l3_filtered[0][0]
        l3_best_prob = sorted_l3_filtered[0][1]

        if l2_best_lang == l3_best_lang:
            # L2 and L3 AGREE → use that language, confidence = L3 %
            final_prediction = l2_best_lang
            final_confidence = round(l3_best_prob, 3)
        else:
            # L2 and L3 DISAGREE → use L2's pick, confidence = L2% + L3% for L2's language
            final_prediction = l2_best_lang
            l3_conf_for_l2_pick = layer3_all_probs.get(l2_best_lang, 0.0)
            final_confidence = round(l2_best_prob + l3_conf_for_l2_pick, 3)

    return {
        "layer1_script":      detected_script,
        "layer1_candidates":  script_candidates,
        "layer2_top3":        layer2_top3_display,
        "layer3_top3":        layer3_top3_display,
        "final_prediction":   final_prediction,
        "final_confidence":   final_confidence,
    }


# ---------------------------------------------------------------------------
# Romanized pipeline — skips Layer 1 entirely
# ---------------------------------------------------------------------------

def predict_romanized(text: str) -> dict:
    """
    Separate pipeline for Romanized (Latin-script) Indic input.

    1. Transliterate the text into ALL supported Indic scripts.
    2. For each transliteration, run Layer 2 and Layer 3 with ALL languages
       as candidates (no script-based filtering).
    3. Track the best L2 and L3 score seen for each language.
    4. Compute final_score = (best_l2 + best_l3) / 2 per language.
    5. Return top-3 for L2, top-3 for L3, and the overall best prediction.
    """
    transliterations = get_transliterations(text)

    # Accumulators: best score seen for each language across all scripts
    best_l2_per_lang = {}  # lang -> float
    best_l3_per_lang = {}  # lang -> float

    # Also run on the original romanized text itself (for the transformer)
    all_variants = dict(transliterations)
    all_variants["Latin (original)"] = text

    for script_label, variant_text in all_variants.items():
        try:
            # --- Layer 2: Transformer (all languages, no filtering) ---
            l2_probs = global_model_loader.predict_layer2(variant_text, ALL_LANGUAGES)

            # Apply Hindi/Bhojpuri rule
            if "Hindi" in l2_probs and "Bhojpuri" in l2_probs:
                hindi_prob = l2_probs["Hindi"]
                bhoj_prob  = l2_probs["Bhojpuri"]
                if abs(hindi_prob - bhoj_prob) < 0.05:
                    shift_amount = hindi_prob * 0.60
                    l2_probs["Hindi"]    -= shift_amount
                    l2_probs["Bhojpuri"] += shift_amount

            for lang, prob in l2_probs.items():
                if lang not in best_l2_per_lang or prob > best_l2_per_lang[lang]:
                    best_l2_per_lang[lang] = prob

            # --- Layer 3: Phoneme model (all languages, no filtering) ---
            phonemes = text_to_phonemes(variant_text)
            l3_probs = global_model_loader.predict_layer3(phonemes, ALL_LANGUAGES)

            for lang, prob in l3_probs.items():
                if lang not in best_l3_per_lang or prob > best_l3_per_lang[lang]:
                    best_l3_per_lang[lang] = prob

        except Exception as e:
            print(f"[predictor] Romanized pipeline error for {script_label}: {e}")

    # --- Build top-3 lists ---
    sorted_l2 = sorted(best_l2_per_lang.items(), key=lambda x: x[1], reverse=True)
    layer2_top3 = [[lang, round(prob, 3)] for lang, prob in sorted_l2]

    sorted_l3 = sorted(best_l3_per_lang.items(), key=lambda x: x[1], reverse=True)
    layer3_top3 = [[lang, round(prob, 3)] for lang, prob in sorted_l3]

    # --- Final prediction: (best_l2 + best_l3) / 2 per language ---
    final_prediction = None
    final_confidence = 0.0

    return {
        "layer1_script": "Skipped (Romanized Input)",
        "layer1_candidates": [lang for lang in ALL_LANGUAGES if lang != "English"],
        "layer2_top3": layer2_top3,
        "layer3_top3": layer3_top3,
        "final_prediction": final_prediction,
        "final_confidence": final_confidence,
        "romanized_mode": True,
        "used_transliteration": "All Scripts",
    }


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def predict_language(text: str) -> dict:
    """
    Routes input to the correct pipeline:
      - Latin-dominant text  → predict_romanized()  (Layer 1 skipped)
      - Native-script text   → _run_pipeline()      (original 3-layer)
    """
    if not text.strip():
        return {
            "layer1_script":       "None",
            "layer1_candidates":   [],
            "layer2_top3":         [],
            "layer3_top3":         [],
            "final_prediction":    None,
            "final_confidence":    0.0,
            "romanized_mode":      False,
            "used_transliteration": None,
        }

    # -----------------------------------------------------------------------
    # ROMANIZED PATH — separate pipeline, Layer 1 skipped
    # -----------------------------------------------------------------------
    if is_latin_dominant(text):
        return predict_romanized(text)

    # -----------------------------------------------------------------------
    # NORMAL PATH  (non-Latin dominant text) — original pipeline unchanged
    # -----------------------------------------------------------------------
    result = _run_pipeline(text)
    result["romanized_mode"]       = False
    result["used_transliteration"] = None
    return result
