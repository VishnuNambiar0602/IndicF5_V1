import sys
# Configure stdout and stderr to use UTF-8 encoding to avoid Windows charmap crash on Indic scripts
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import torch
# Disable torch.compile on Windows to bypass Triton compilation freeze/delays
torch.compile = lambda model, *args, **kwargs: model

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

# Set HF Token early for authenticated downloads
if "HF_TOKEN" not in os.environ:
    # Fallback to loading from environment or user-defined variable if needed
    pass

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
    print("Indic-FS: Patched f5_tts.infer.utils_infer.load_model successfully.")
except Exception as e:
    print(f"Indic-FS Warning: Failed to patch load_model: {e}")

import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
import torchaudio
import librosa
import numpy as np

# Monkeypatch torchaudio.load and torchaudio.save to avoid TorchCodec dependency issues on some systems
_original_torchaudio_load = torchaudio.load
def patched_torchaudio_load(path, *args, **kwargs):
    try:
        # Try original first, but if it fails with the specific TorchCodec error, use librosa
        return _original_torchaudio_load(path, *args, **kwargs)
    except Exception as e:
        if "TorchCodec" in str(e) or "libtorchcodec" in str(e):
            # print(f"Indic-FS: torchaudio.load failed ({e}), falling back to librosa.")
            y, sr = librosa.load(path, sr=None, mono=False)
            if y.ndim == 1:
                y = y[np.newaxis, :]
            return torch.from_numpy(y), sr
        raise e
torchaudio.load = patched_torchaudio_load

_original_torchaudio_save = torchaudio.save
def patched_torchaudio_save(path, src, sample_rate, *args, **kwargs):
    try:
        return _original_torchaudio_save(path, src, sample_rate, *args, **kwargs)
    except Exception as e:
        if "TorchCodec" in str(e) or "libtorchcodec" in str(e):
            import soundfile as sf
            # Convert torch tensor to numpy
            if hasattr(src, "cpu"):
                data = src.cpu().numpy()
            else:
                data = src
            # soundfile expects (samples, channels) but torchaudio uses (channels, samples)
            if data.ndim == 2:
                data = data.T
            sf.write(path, data, sample_rate)
            return
        raise e
torchaudio.save = patched_torchaudio_save

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Indic-FS: Using device {device} for model inference.")

# Try importing IndicTranslator from IndicTransToolkit, or define a fallback wrapper
try:
    from IndicTransToolkit import IndicTranslator
    print("Indic-FS: Imported IndicTranslator from IndicTransToolkit.")
except ImportError:
    print("Indic-FS: IndicTranslator not found in IndicTransToolkit. Using custom fallback wrapper.")
    
    try:
        from IndicTransToolkit.processor import IndicProcessor
    except ImportError:
        try:
            from IndicTransToolkit import IndicProcessor
        except ImportError:
            IndicProcessor = None
            print("Indic-FS: WARNING: IndicProcessor could not be imported. Preprocessing may be limited.")

    class IndicTranslator:
        def __init__(self, ckpt_dir: str, device: str = "cpu"):
            self.device = device
            self.ckpt_dir = ckpt_dir
            self.model_type = "indic-indic" if "indic-indic" in ckpt_dir else "en-indic"
            
            print(f"Indic-FS Fallback: Loading tokenizer for {ckpt_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
            print(f"Indic-FS Fallback: Loading seq2seq model for {ckpt_dir}...")
            
            # Use float16 on CUDA to save memory, float32 on CPU
            dtype = torch.float16 if "cuda" in device else torch.float32
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                ckpt_dir, 
                trust_remote_code=True,
                torch_dtype=dtype
            ).to(self.device)
            
            if IndicProcessor is not None:
                self.ip = IndicProcessor(inference=True)
            else:
                self.ip = None
            print(f"Indic-FS Fallback: Model loaded successfully on {device}.")

        def batch_translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
            if not texts:
                return []
                
            # Check if we are trying to translate from English using an indic-indic model
            if src_lang == "eng_Latn" and self.model_type == "indic-indic":
                print("Indic-FS Warning: English detected but only indic-indic model loaded. Translation may be poor.")

            if self.ip is not None:
                try:
                    batch = self.ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
                except Exception as e:
                    print(f"Indic-FS Translation Preprocess Error: {e}. Using raw text.")
                    batch = texts
            else:
                batch = texts

            inputs = self.tokenizer(
                batch, 
                truncation=True, 
                padding="longest", 
                return_tensors="pt"
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, 
                    num_beams=5, 
                    max_length=256
                )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            if self.ip is not None:
                try:
                    decoded = self.ip.postprocess_batch(decoded, lang=tgt_lang)
                except Exception as e:
                    print(f"Indic-FS Translation Postprocess Error: {e}")
                
            return decoded

# --- Lazy Loaders ---

_indicf5_instance = None
_translator_indic_instance = None
_translator_en_instance = None
_multi_translator_instance = None

# Shared NLLB translator instance
nllb_translator_instance = None

_xtts_instance = None

def get_xtts():
    global _xtts_instance
    if _xtts_instance is None:
        print("Indic-FS: Loading XTTS-v2 voice cloning model...")
        from TTS.api import TTS
        _xtts_instance = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("Indic-FS: XTTS-v2 loaded successfully.")
    return _xtts_instance

def get_nllb_fallback():
    global nllb_translator_instance
    if nllb_translator_instance is None:
        print("Indic-FS: Attempting to load open-source fallback translator 'facebook/nllb-200-distilled-600M'...")
        try:
            class NllbTranslatorFallback:
                def __init__(self, model_name="facebook/nllb-200-distilled-600M", device="cpu"):
                    self.device = device
                    print(f"Indic-FS Fallback: Loading tokenizer for {model_name}...")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    print(f"Indic-FS Fallback: Loading seq2seq model for {model_name}...")
                    dtype = torch.float16 if "cuda" in device else torch.float32
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
                    print(f"Indic-FS Fallback: NLLB translator loaded successfully on {device}.")
                    
                def batch_translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
                    if not texts:
                        return []
                    translated_texts = []
                    for text in texts:
                        self.tokenizer.src_lang = src_lang
                        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                        with torch.inference_mode():
                            translated_tokens = self.model.generate(
                                **inputs,
                                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
                                max_length=256
                            )
                        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                        translated_texts.append(decoded)
                    return translated_texts
            
            nllb_translator_instance = NllbTranslatorFallback(device="cpu")
        except Exception as err:
            print(f"Indic-FS Error: Failed to load NLLB fallback translator: {err}")
    return nllb_translator_instance

def get_translator_indic():
    global _translator_indic_instance
    if _translator_indic_instance is None:
        try:
            print("Indic-FS: Loading Indic-to-Indic translator on CPU...")
            _translator_indic_instance = IndicTranslator(ckpt_dir="ai4bharat/indictrans2-indic-indic-1B", device="cpu")
        except Exception as e:
            print(f"Indic-FS Warning: Failed to load Indic-to-Indic translator: {e}")
            _translator_indic_instance = get_nllb_fallback()
    return _translator_indic_instance

def get_translator_en():
    global _translator_en_instance
    if _translator_en_instance is None:
        try:
            print("Indic-FS: Loading English-to-Indic translator on CPU...")
            _translator_en_instance = IndicTranslator(ckpt_dir="ai4bharat/indictrans2-en-indic-1B", device="cpu")
        except Exception as e:
            print(f"Indic-FS Warning: Failed to load English-to-Indic translator: {e}")
            _translator_en_instance = get_nllb_fallback()
    return _translator_en_instance

class MultiTranslator:
    def batch_translate(self, texts, src_lang, tgt_lang):
        if src_lang == "eng_Latn":
            t_en = get_translator_en()
            if t_en:
                return t_en.batch_translate(texts, src_lang, tgt_lang)
            else:
                print("Indic-FS Error: English source detected but en-indic model not loaded.")
                return texts
        else:
            t_indic = get_translator_indic()
            if t_indic:
                return t_indic.batch_translate(texts, src_lang, tgt_lang)
            else:
                print("Indic-FS Error: Indic source detected but indic-indic model not loaded.")
                return texts

def get_multi_translator():
    global _multi_translator_instance
    if _multi_translator_instance is None:
        _multi_translator_instance = MultiTranslator()
    return _multi_translator_instance

def __getattr__(name):
    if name == "xtts":
        return get_xtts()
    elif name == "translator":
        return get_multi_translator()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# ASR (Whisper) auto-transcriber
asr_pipeline = None

def get_asr_pipeline():
    global asr_pipeline
    if asr_pipeline is None:
        print("Indic-FS: Loading Whisper-large-v3 ASR pipeline for auto-transcription...")
        try:
            from transformers import pipeline
            # Use GPU if available, else CPU
            device_idx = 0 if "cuda" in device else -1
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                device=device_idx
            )
            print("Indic-FS: Whisper ASR pipeline loaded successfully.")
        except Exception as e:
            print(f"Indic-FS Warning: Failed to load Whisper ASR pipeline: {e}")
    return asr_pipeline

def auto_transcribe(audio_path: str, language: str = None) -> str:
    pipe = get_asr_pipeline()
    if pipe is None:
        return ""
    try:
        print(f"Indic-FS ASR: Transcribing {audio_path}...")
        kwargs = {}
        if language:
            kwargs["generate_kwargs"] = {"language": language}
        result = pipe(audio_path, **kwargs)
        transcript = result.get("text", "").strip()
        print(f"Indic-FS ASR: Transcribed text: '{transcript}'")
        return transcript
    except Exception as e:
        print(f"Indic-FS ASR Warning: Auto-transcription failed: {e}")
        return ""

