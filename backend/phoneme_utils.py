import json
import os
import torch

# Resolve paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vocab_path = os.path.join(PROJECT_ROOT, "models", "layer3model", "phoneme_vocab.json")

try:
    with open(vocab_path, "r", encoding="utf-8") as f:
        phoneme_vocab = json.load(f)
    print(f"Phoneme vocab loaded: {len(phoneme_vocab)} entries")
except Exception as e:
    print(f"Warning: Could not load phoneme vocab from {vocab_path}: {e}")
    phoneme_vocab = {"<PAD>": 0, "<UNK>": 1}

def text_to_phonemes(text: str) -> str:
    """
    Generate phonemes from text.
    Falls back to raw characters if phonemizer/espeak is not installed.
    """
    try:
        from phonemizer import phonemize
        ph_text = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=False)
        return ph_text
    except Exception as e:
        print(f"Phonemizer fallback (espeak may not be installed): {e}")
        # Fallback: return the raw text characters as pseudo-phonemes
        return text
        
def tokenize_phonemes(phoneme_string: str, max_len: int = 65) -> torch.Tensor:
    """Convert a phoneme string into a padded tensor of vocabulary indices."""
    indices = []
    i = 0
    while i < len(phoneme_string):
        # Try 2-char phonemes first (e.g. 'aː', 'dʰ')
        if i + 1 < len(phoneme_string) and phoneme_string[i:i+2] in phoneme_vocab:
            indices.append(phoneme_vocab[phoneme_string[i:i+2]])
            i += 2
        elif phoneme_string[i] in phoneme_vocab:
            indices.append(phoneme_vocab[phoneme_string[i]])
            i += 1
        elif phoneme_string[i] == ' ':
            i += 1  # skip spaces
        else:
            indices.append(phoneme_vocab.get("<UNK>", 1))
            i += 1

    # Truncate or pad to max_len
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        pad_id = phoneme_vocab.get("<PAD>", 0)
        indices = indices + [pad_id] * (max_len - len(indices))
        
    return torch.tensor([indices], dtype=torch.long)
