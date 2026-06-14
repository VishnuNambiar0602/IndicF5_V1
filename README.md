# Indic-FS — AI Cross-Lingual Voice Cloning System

Indic-FS is a full-stack web application that allows users to record or upload a sample of their voice once and synthesize high-fidelity speech in 11 different Indian languages using their cloned voice. Leveraging advanced deep learning models, it bridges linguistic barriers across India by keeping the speaker's unique vocal identity consistent.

---

## Architecture Diagram

```
           +-------------------------------------------------+
           |          Frontend (HTML5 + Three.js 3D)         |
           +-------------------------------------------------+
              | (Upload Audio)                     ^ (Play / Download Audio)
              v                                    |
  +-----------------------+            +-----------------------+
  |  POST /upload-voice   |            |    POST /synthesize   |
  +-----------------------+            +-----------------------+
              |                                    |
              | (Save to uploads/)                 | (Invoke ML Pipeline)
              v                                    v
     +-----------------+                  +-----------------+
     |    utils.py     |                  |   pipeline.py   |
     +-----------------+                  +-----------------+
      - Audio load/save                    - Script detector
      - 3s Min Validation                  - Text Translation
      - Resampling/Mono                    - Speech Synthesis
                                            /             \
                                           v               v
                                   +---------------+ +---------------+
                                   |  IndicTrans2  | |   IndicF5     |
                                   |  NMT Model    | |  TTS Model    |
                                   +---------------+ +---------------+
                                           \               /
                                            v             v
                                           (Outputs Saved to outputs/)
```

---

## Setup Instructions

### 1. Clone the repository and navigate inside
```bash
git clone <repository_url>
cd indic-fs
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install IndicTrans2 inference toolkit
You need to install the `IndicTransToolkit` / inference engine from the official repository:
```bash
git clone https://github.com/AI4Bharat/IndicTrans2.git
cd IndicTrans2/inference
pip install -e .
cd ../..
```

### 4. Run the FastAPI backend server
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
Open your browser and visit [http://localhost:8000](http://localhost:8000) to access the application.

---

## API Documentation

### 1. `POST /upload-voice`
Uploads a reference voice sample from the user, validates that the duration is over 3 seconds, and registers it.
- **Request Type**: Multipart Form Data (`file`)
- **Response**:
  ```json
  {
    "voice_id": "8bfa2e9a-4c28-4bc2-9e90-258079a4de54",
    "filename": "uploads/ref_8bfa2e9a-4c28-4bc2-9e90-258079a4de54.wav",
    "duration": 5.43
  }
  ```

### 2. `POST /synthesize`
Translates and synthesizes input text into a cloned voice in multiple target languages.
- **Request Type**: JSON Body
  ```json
  {
    "voice_id": "8bfa2e9a-4c28-4bc2-9e90-258079a4de54",
    "text": "नमस्ते, आप कैसे हैं?",
    "target_langs": ["hi", "ta", "te"],
    "source_lang": "hi" // optional, auto-detected if omitted
  }
  ```
- **Response**:
  ```json
  {
    "outputs": {
      "hi": "/download/output_hi_2e9afc28.wav",
      "ta": "/download/output_ta_4d98ab32.wav",
      "te": "/download/output_te_1f23cd89.wav"
    }
  }
  ```

### 3. `GET /download/{filename}`
Downloads the generated/uploaded WAV audio file from the server.
- **Parameters**: `filename` (e.g. `output_hi_2e9afc28.wav`)
- **Response**: Audio file download stream (WAV)

### 4. `GET /health`
Performs checks on backend models and environment.
- **Response**:
  ```json
  {
    "status": "ok",
    "models_loaded": true
  }
  ```

### 5. `GET /`
Serves the 3D-interactive single-page web dashboard interface.

---

## Supported Languages

| Code | Language | Native Script |
| --- | --- | --- |
| `hi` | Hindi | हिंदी |
| `ta` | Tamil | தமிழ் |
| `te` | Telugu | తెలుగు |
| `kn` | Kannada | ಕನ್ನಡ |
| `ml` | Malayalam | മലയാളം |
| `bn` | Bengali | বাংলা |
| `mr` | Marathi | मराठी |
| `gu` | Gujarati | ગુજરાતી |
| `pa` | Punjabi | ਪੰਜਾਬੀ |
| `or` | Odia | ଓଡ଼ିଆ |
| `as` | Assamese | অসমীয়া |

---

## Known Issues & Fixes

1. **Transformers Compatibility**:
   `IndicF5` is built on F5-TTS, which depends strictly on `transformers==4.49.0`. Standard environments downloading a newer version will throw module-attribute errors. Ensure you keep it pinned.

2. **CUDA vs CPU**:
   For acceptable speed during synthesis, a GPU is recommended. The backend will automatically fall back to CPU if CUDA is not available. If you see high latency, please check that PyTorch detects your GPU by running:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

3. **Hugging Face Hub Authentication**:
   Downloading these models requires a Hugging Face Token. The application is pre-configured to use the developer token provided, but you can override it in `models.py` or by setting the `HF_TOKEN` environment variable.
