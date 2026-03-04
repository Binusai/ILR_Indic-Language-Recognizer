import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .phoneme_model import PhonemeLID
from .phoneme_utils import tokenize_phonemes
import json

# Resolve model paths relative to project root (ILR directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ModelLoader:
    def __init__(self):
        self.layer2_path = os.path.join(PROJECT_ROOT, "models", "layer2model")
        self.layer3_path = os.path.join(PROJECT_ROOT, "models", "layer3model")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.layer2_model = None
        self.layer2_tokenizer = None
        self.layer3_model = None
        self.layer3_labels = []
        
        self.is_loaded = False
        self.load_models()

    def load_models(self):
        """Loads Layer 2 HuggingFace model and Layer 3 PyTorch model from disk."""
        # --- LAYER 2 ---
        try:
            print(f"Loading Layer 2 transformer from: {self.layer2_path}")
            self.layer2_tokenizer = AutoTokenizer.from_pretrained(self.layer2_path)
            self.layer2_model = AutoModelForSequenceClassification.from_pretrained(self.layer2_path)
            self.layer2_model.to(self.device)
            self.layer2_model.eval()
            print("Layer 2 model loaded successfully.")
        except Exception as e:
            print(f"Error loading Layer 2 model: {e}")

        # --- LAYER 3 ---
        try:
            print(f"Loading Layer 3 phoneme model from: {self.layer3_path}")
            l3_config_path = os.path.join(self.layer3_path, "model_config.json")
            with open(l3_config_path, "r", encoding="utf-8") as f:
                l3_config = json.load(f)
                
            self.layer3_model = PhonemeLID(
                vocab_size=l3_config.get("vocab_size", 132),
                embed_dim=l3_config.get("embed_dim", 128),
                hidden_dim=l3_config.get("hidden_dim", 256),
                num_labels=l3_config.get("num_labels", 15)
            )
            
            model_pt_path = os.path.join(self.layer3_path, "layer3_best.pt")
            self.layer3_model.load_state_dict(
                torch.load(model_pt_path, map_location=self.device, weights_only=True)
            )
            self.layer3_model.to(self.device)
            self.layer3_model.eval()
            
            # Load L3 labels
            l3_labels_path = os.path.join(self.layer3_path, "label_mapping.json")
            with open(l3_labels_path, "r", encoding="utf-8") as f:
                l3_mapping = json.load(f)
                idx_to_label = {int(k): v for k, v in l3_mapping["id2label"].items()}
                self.layer3_labels = [idx_to_label[i] for i in range(len(idx_to_label))]
                
            print("Layer 3 model loaded successfully.")
        except Exception as e:
            print(f"Error loading Layer 3 model: {e}")
            
        self.is_loaded = True
        
    def predict_layer2(self, text: str, candidates: list) -> dict:
        """Layer 2 inference using HuggingFace BERT transformer."""
        if not self.layer2_model or not self.layer2_tokenizer:
            print("Layer 2 model not loaded. Returning uniform distribution.")
            return {c: 1.0 / len(candidates) for c in candidates} if candidates else {"Unknown": 1.0}

        with torch.no_grad():
            inputs = self.layer2_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.layer2_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
            
        # id2label keys are strings like "0", "1", ... in config
        id2label = self.layer2_model.config.id2label
        
        filtered_probs = {}
        for idx, prob in enumerate(probs):
            lang = id2label.get(str(idx), id2label.get(idx, f"Unknown_{idx}"))
            if not candidates or lang in candidates:
                filtered_probs[lang] = float(prob)

        # If script detection returned "Unknown", pass all languages through
        if "Unknown" in candidates or not filtered_probs:
            filtered_probs = {}
            for idx, prob in enumerate(probs):
                lang = id2label.get(str(idx), id2label.get(idx, f"Unknown_{idx}"))
                filtered_probs[lang] = float(prob)
                
        # Normalize
        total_prob = sum(filtered_probs.values())
        if total_prob > 0:
            filtered_probs = {k: v / total_prob for k, v in filtered_probs.items()}
                
        return filtered_probs

    def predict_layer3(self, phoneme_string: str, candidates: list) -> dict:
        """Layer 3 Phoneme BiLSTM inference."""
        if not self.layer3_model:
            print("Layer 3 model not loaded. Returning uniform distribution.")
            return {c: 1.0 / len(candidates) for c in candidates[:2]} if candidates else {}

        input_tensor = tokenize_phonemes(phoneme_string).to(self.device)
        
        with torch.no_grad():
            logits = self.layer3_model(input_tensor)
            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
            
        filtered_probs = {}
        if probs.ndim == 0:
            probs = [float(probs)]

        for idx, prob in enumerate(probs):
            if idx < len(self.layer3_labels):
                lang = self.layer3_labels[idx]
                if lang in candidates:
                    filtered_probs[lang] = float(prob)
                    
        total_prob = sum(filtered_probs.values())
        if total_prob > 0:
            filtered_probs = {k: v / total_prob for k, v in filtered_probs.items()}
        elif candidates:
            filtered_probs = {c: 1.0 / len(candidates) for c in candidates}
                
        return filtered_probs

global_model_loader = ModelLoader()
