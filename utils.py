import os
from uuid import uuid4
import librosa
import soundfile as sf
import numpy as np

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file, resample to 16000 Hz, and convert to mono.
    Returns the normalized float32 numpy array and the sample rate (16000).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    # librosa.load automatically resamples to sr (16000) and converts to mono (mono=True)
    # It also normalizes to float32 in [-1.0, 1.0] by default.
    y, sr = librosa.load(path, sr=16000, mono=True)
    return y.astype(np.float32), sr

def save_audio(numpy_array: np.ndarray, sample_rate: int, output_path: str):
    """
    Save a float32 numpy array as a WAV file using soundfile.
    """
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, numpy_array, sample_rate)

def trim_audio(numpy_array: np.ndarray, sample_rate: int, max_seconds: float = 20.0) -> np.ndarray:
    """
    Trim audio array to a maximum duration of max_seconds.
    """
    max_samples = int(max_seconds * sample_rate)
    return numpy_array[:max_samples]

def validate_audio(path: str):
    """
    Validate that the audio file exists, has a supported extension,
    and has a duration greater than 3 seconds.
    Raises ValueError with a clear message if invalid.
    """
    if not os.path.exists(path):
        raise ValueError(f"Audio file does not exist at: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    allowed_extensions = {".wav", ".mp3", ".ogg", ".m4a", ".webm"}
    if ext not in allowed_extensions:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: {', '.join(allowed_extensions)}")
    
    try:
        # Get duration robustly
        try:
            duration = librosa.get_duration(path=path)
        except Exception:
            # Fallback by loading the file metadata/audio
            y, sr = librosa.load(path, sr=None)
            duration = len(y) / sr
    except Exception as e:
        raise ValueError(f"Could not read audio file metadata or file is corrupted: {str(e)}")
    
    if duration < 3.0:
        raise ValueError(f"Reference voice audio must be at least 3 seconds long (currently {duration:.2f} seconds).")

def generate_output_filename(lang_code: str) -> str:
    """
    Generate a unique output filepath for the synthesized language.
    """
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    return f"outputs/output_{lang_code}_{uuid4().hex[:8]}.wav"
