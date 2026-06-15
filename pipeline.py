import os
import numpy as np
import models
import utils
from uuid import uuid4

# Language code mapping from 2-letter code to IndicTrans2 code
LANG_CODE_MAP = {
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng"
}

# Reverse mapping
INV_LANG_CODE_MAP = {v: k for k, v in LANG_CODE_MAP.items()}

XTTS_LANG_MAP = {
    "hin_Deva": "hi",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "kan_Knda": "kn",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "ben_Beng": "bn",
    "pan_Guru": "pa",
    "guj_Gujr": "gu",
    "eng_Latn": "en",
}

def detect_language(text: str) -> str:
    """
    Detect script of text using Unicode ranges and return the IndicTrans2 language code.
    """
    counts = {
        "hin_Deva": 0, # Hindi/Devanagari
        "tam_Taml": 0, # Tamil
        "tel_Telu": 0, # Telugu
        "kan_Knda": 0, # Kannada
        "mal_Mlym": 0, # Malayalam
        "ben_Beng": 0, # Bengali/Assamese
        "guj_Gujr": 0, # Gujarati
        "pan_Guru": 0, # Gurmukhi (Punjabi)
        "ory_Orya": 0, # Odia
        "eng_Latn": 0, # English/Latin
    }
    for char in text:
        cp = ord(char)
        if 0x0900 <= cp <= 0x097F:
            counts["hin_Deva"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["tam_Taml"] += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            counts["tel_Telu"] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            counts["kan_Knda"] += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            counts["mal_Mlym"] += 1
        elif 0x0980 <= cp <= 0x09FF:
            counts["ben_Beng"] += 1
        elif 0x0A80 <= cp <= 0x0AFF:
            counts["guj_Gujr"] += 1
        elif 0x0A00 <= cp <= 0x0A7F:
            counts["pan_Guru"] += 1
        elif 0x0B00 <= cp <= 0x0B7F:
            counts["ory_Orya"] += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            counts["eng_Latn"] += 1

    max_lang = max(counts, key=counts.get)
    if counts[max_lang] == 0:
        return "hin_Deva"  # Fallback to Hindi Devanagari
    return max_lang

def translate_google(text: str, target_lang: str) -> str:
    """
    Translate text using the free Google Translate API.
    """
    import urllib.parse
    import urllib.request
    import json
    try:
        quoted = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_lang}&dt=t&q={quoted}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            translation = "".join([segment[0] for segment in data[0] if segment[0]])
            return translation
    except Exception as e:
        print(f"Indic-FS: Google Translate API failed ({e}).")
        return None

def clone_and_synthesize(
    ref_audio_path: str, 
    input_text: str, 
    target_lang_codes: list[str],
    source_lang: str = None,
    ref_text: str = "",
    cloning_method: str = "openvoice"
) -> dict:
    """
    Clone voice from ref_audio and synthesize input_text into multiple target languages.
    """
    # 1. Resolve source language
    if not source_lang:
        src_lang_it = detect_language(input_text)
    else:
        src_lang_it = LANG_CODE_MAP.get(source_lang, source_lang)
    
    print(f"Indic-FS Pipeline: Source language resolved as {src_lang_it}")
    
    # 2. Validate reference audio file
    utils.validate_audio(ref_audio_path)
    
    results = {}
    
    # Improved reference text handling with auto-transcription fallback
    ref_text_val = ref_text.strip() if ref_text else ""
    
    generic_placeholders = {
        "नमस्ते, यह मेरे आवाज का नमूना है।",
        "नमस्ते, यह मेरे आवाज का नमूना है",
        "Hello, this is a sample of my voice.",
        "Hello, this is a sample of my voice"
    }
    
    if not ref_text_val or ref_text_val in generic_placeholders:
        print(f"Indic-FS Pipeline: Reference transcript is empty or placeholder ('{ref_text_val}'). Auto-transcribing reference voice...")
        try:
            whisper_lang = INV_LANG_CODE_MAP.get(src_lang_it, None)
            if src_lang_it == "eng_Latn":
                whisper_lang = "en"
            auto_transcript = models.auto_transcribe(ref_audio_path, language=whisper_lang)
            if auto_transcript:
                ref_text_val = auto_transcript
                print(f"Indic-FS Pipeline: Successfully auto-transcribed reference audio: '{ref_text_val}'")
            else:
                raise ValueError("ASR output was empty.")
        except Exception as asr_err:
            ref_text_val = ""  # Better to pass empty than a wrong language placeholder
            print(f"Indic-FS Pipeline: Auto-transcription failed ({asr_err}). Passing empty ref_text.")
    else:
        print(f"Indic-FS Pipeline: Using user-provided reference transcript: '{ref_text_val}'")
            
    print(f"Indic-FS Pipeline: Using reference text: '{ref_text_val}'")
    
    for tgt in target_lang_codes:
        # Convert target code to full IndicTrans2 language code if it's 2-letter
        tgt_lang_it = LANG_CODE_MAP.get(tgt, tgt)
        key_code = INV_LANG_CODE_MAP.get(tgt_lang_it, tgt)
        
        print(f"Indic-FS Pipeline: Processing for '{tgt_lang_it}' ({key_code})...")
        
        try:
            # a. Translate text if source language is different from target
            if src_lang_it == tgt_lang_it:
                translated_text = input_text
                print(f"Indic-FS Pipeline: Source and target match. No translation needed.")
            else:
                print(f"Indic-FS Pipeline: Attempting online translation to '{tgt}'...")
                translated_text = translate_google(input_text, tgt)
                if translated_text:
                    print(f"Indic-FS Pipeline: Online translation succeeded: '{translated_text}'")
                else:
                    print(f"Indic-FS Pipeline: Online translation failed/offline. Falling back to local translator...")
                    print(f"Indic-FS Pipeline: Translating locally from {src_lang_it} to {tgt_lang_it}...")
                    translations = models.translator.batch_translate([input_text], src_lang_it, tgt_lang_it)
                    if not translations:
                        raise ValueError(f"Translation failed: empty response.")
                    translated_text = translations[0]
                    print(f"Indic-FS Pipeline: Local translation: '{translated_text}'")
            
            # b. Synthesize speech
            xtts_lang = XTTS_LANG_MAP.get(tgt_lang_it, "hi")
            print(f"Indic-FS Pipeline: Selecting TTS engine for '{xtts_lang}'...")
            
            XTTS_SUPPORTED = {"hi", "en"}
            
            use_xtts = False
            if cloning_method == "xtts" and xtts_lang in XTTS_SUPPORTED:
                use_xtts = True
                
            if use_xtts:
                print(f"Indic-FS Pipeline: Using XTTS-v2 for '{xtts_lang}'...")
                xtts = models.xtts
                wav = xtts.tts(
                    text=translated_text,
                    speaker_wav=ref_audio_path,
                    language=xtts_lang
                )
                audio_out = np.array(wav, dtype=np.float32)
                out_sr = 24000
                
                # Post-processing synthesis output
                if hasattr(audio_out, "cpu"):
                    audio_out = audio_out.cpu().numpy()
                
                if isinstance(audio_out, (tuple, list)):
                    audio_out = audio_out[0]
                    
                if audio_out.dtype == np.int16:
                    audio_out = audio_out.astype(np.float32) / 32768.0
                elif audio_out.dtype == np.int32:
                    audio_out = audio_out.astype(np.float32) / 2147483648.0
                    
                # c. Save audio
                output_path = utils.generate_output_filename(key_code)
                utils.save_audio(audio_out, out_sr, output_path)
                print(f"Indic-FS Pipeline: Saved to {output_path}")
            else:
                vc_method = "openvoice" if cloning_method in ("openvoice", "xtts") else "freevc"
                print(f"Indic-FS Pipeline: Using MMS-TTS + {vc_method.upper()} Voice Conversion for '{xtts_lang}'...")
                # 1. Synthesize base speech using MMS-TTS
                audio_out, out_sr = models.synthesize_mms(translated_text, xtts_lang)
                
                # Post-processing synthesis output
                if hasattr(audio_out, "cpu"):
                    audio_out = audio_out.cpu().numpy()
                
                if isinstance(audio_out, (tuple, list)):
                    audio_out = audio_out[0]
                    
                if audio_out.dtype == np.int16:
                    audio_out = audio_out.astype(np.float32) / 32768.0
                elif audio_out.dtype == np.int32:
                    audio_out = audio_out.astype(np.float32) / 2147483648.0
                
                # 2. Save base speech to a temporary file
                temp_mms_path = f"outputs/temp_mms_{key_code}_{uuid4().hex[:8]}.wav"
                utils.save_audio(audio_out, out_sr, temp_mms_path)
                
                # 3. Perform voice conversion
                output_path = utils.generate_output_filename(key_code)
                models.voice_conversion(
                    source_wav_path=temp_mms_path,
                    target_wav_path=ref_audio_path,
                    output_wav_path=output_path,
                    method=vc_method
                )
                
                # 4. Clean up the temporary MMS file
                try:
                    if os.path.exists(temp_mms_path):
                        os.remove(temp_mms_path)
                except Exception as cleanup_err:
                    print(f"Indic-FS Warning: Failed to delete temporary MMS file {temp_mms_path}: {cleanup_err}")
                
                print(f"Indic-FS Pipeline: Voice-converted file saved to {output_path}")
            
            results[key_code] = output_path
            
        except Exception as e:
            print(f"Indic-FS Pipeline: Error for {tgt}: {str(e)}")
            results[key_code] = f"ERROR: {str(e)}"
    
    return results
