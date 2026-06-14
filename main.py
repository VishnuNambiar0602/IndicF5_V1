import sys
# Configure stdout and stderr to use UTF-8 encoding to avoid Windows charmap crash on Indic scripts
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import importlib

# Python 3.14 compatibility hotfix for protobuf metaclass crash
_original_import_module = importlib.import_module
def _custom_import_module(name, package=None):
    if name == 'google._upb._message':
        raise ImportError("Bypassing google._upb._message under Python 3.14")
    return _original_import_module(name, package)
importlib.import_module = _custom_import_module

import os
# Force pure python implementation for protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Disable Xet backend and hf_transfer to avoid socket freeze/crash issues on large files
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# F5-TTS load_model checkpoint path monkeypatch
try:
    import f5_tts.infer.utils_infer as f5_infer
    _original_load_model = f5_infer.load_model
    def patched_load_model(model_cls, model_cfg, ckpt_path=None, *args, **kwargs):
        if ckpt_path is None or ckpt_path == "":
            original_load_checkpoint = f5_infer.load_checkpoint
            f5_infer.load_checkpoint = lambda model, *a, **k: model
            try:
                model = _original_load_model(model_cls, model_cfg, "", *args, **kwargs)
            finally:
                f5_infer.load_checkpoint = original_load_checkpoint
            return model
        return _original_load_model(model_cls, model_cfg, ckpt_path, *args, **kwargs)
    f5_infer.load_model = patched_load_model
    print("Indic-FS: Patched f5_tts.infer.utils_infer.load_model successfully in main.py.")
except Exception as e:
    print(f"Indic-FS Warning: Failed to patch load_model in main.py: {e}")

import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import librosa

import utils
import pipeline

app = FastAPI(title="Indic-FS Voice Cloning System")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory mapping from voice_id to local filepath
voice_store = {}

# Ensure required directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

class SynthesizeRequest(BaseModel):
    voice_id: str
    text: str
    target_langs: list[str]
    source_lang: str = None
    ref_text: str = ""

@app.post("/upload-voice")
async def upload_voice(file: UploadFile = File(...)):
    try:
        # Generate unique voice ID
        voice_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1].lower()
        if not ext:
            ext = ".wav"
            
        filepath = f"uploads/ref_{voice_id}{ext}"
        
        # Save file to disk
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Validate audio properties
        try:
            utils.validate_audio(filepath)
        except ValueError as val_err:
            if os.path.exists(filepath):
                os.remove(filepath)
            raise HTTPException(status_code=400, detail=str(val_err))
            
        # Get duration
        try:
            duration = float(librosa.get_duration(path=filepath))
        except Exception:
            y, sr = librosa.load(filepath, sr=None)
            duration = float(len(y) / sr)
            
        # Record voice path
        voice_store[voice_id] = filepath
        
        return {
            "voice_id": voice_id,
            "filename": filepath,
            "duration": round(duration, 2)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice upload failed: {str(e)}")

@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    ref_audio_path = voice_store.get(req.voice_id)
    if not ref_audio_path or not os.path.exists(ref_audio_path):
        raise HTTPException(status_code=404, detail="Voice ID not found or reference file missing.")
        
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text to synthesize cannot be empty.")
        
    if not req.target_langs:
        raise HTTPException(status_code=400, detail="At least one target language must be specified.")
        
    try:
        # Synthesize audio files
        output_results = pipeline.clone_and_synthesize(
            ref_audio_path=ref_audio_path,
            input_text=req.text,
            target_lang_codes=req.target_langs,
            source_lang=req.source_lang,
            ref_text=req.ref_text
        )
        
        # Format paths as downloadable URLs
        outputs_urls = {}
        for lang, path in output_results.items():
            if path.startswith("ERROR:"):
                outputs_urls[lang] = path
            else:
                filename = os.path.basename(path)
                outputs_urls[lang] = f"/download/{filename}"
                
        return {"outputs": outputs_urls}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice synthesis pipeline error: {str(e)}")

@app.get("/download/{filename}")
def download_file(filename: str):
    # Secure filename against path traversal
    safe_filename = os.path.basename(filename)
    filepath = os.path.join("outputs", safe_filename)
    
    if not os.path.exists(filepath):
        # Check uploads directory as fallback
        filepath_upload = os.path.join("uploads", safe_filename)
        if os.path.exists(filepath_upload):
            return FileResponse(filepath_upload)
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(filepath)

@app.get("/health")
def health():
    import models
    models_loaded = (getattr(models, "_indicf5_instance", None) is not None) and (getattr(models, "_multi_translator_instance", None) is not None)
    return {
        "status": "ok",
        "models_loaded": models_loaded
    }

@app.get("/")
def serve_frontend():
    frontend_path = "frontend/index.html"
    if not os.path.exists(frontend_path):
        raise HTTPException(status_code=404, detail="Frontend file index.html not found.")
    return FileResponse(frontend_path)
