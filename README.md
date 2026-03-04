# ILR — Indic Language Recognizer

A multi-layer system for identifying Indic languages using script detection, transformer models, and phoneme analysis.

## Features

* True multilingual text and speech input without bias toward English.
* 3-Layer architecture (Unicode Script Detection -> Transformer Model -> Phoneme Model).
* Resolves Hindi vs Bhojpuri ambiguity.
* Modern Dark UI with glassmorphism and animated components.

## Setup Instructions

### Pre-requisites
* Python 3.9+

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Start the Backend Server
Run the FastAPI application locally:
```bash
uvicorn backend.app:app --reload
```

### 3. Open the Frontend
Simply open `frontend/index.html` in your favorite web browser or use a live server extension (like VSCode Live Server) for hot-reloading.
