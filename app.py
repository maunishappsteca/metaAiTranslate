import os
import json
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import runpod  # Added missing import

# Initialize the translator model (loaded once on cold start)
tokenizer = None
model = None

# Language code mapping (ISO 639-1 to NLLB codes)
LANGUAGE_CODES = {
    "en": "eng_Latn",    # English
    "hi": "hin_Deva",    # Hindi
    "es": "spa_Latn",    # Spanish
    "fr": "fra_Latn",    # French
    "de": "deu_Latn",    # German
    "it": "ita_Latn",    # Italian
    "pt": "por_Latn",    # Portuguese
    "ru": "rus_Cyrl",    # Russian
    "zh": "zho_Hans",    # Chinese
    "ja": "jpn_Jpan",    # Japanese
    "ar": "arb_Arab",    # Arabic
    "he": "heb_Hebr",    # Hebrew
    "ko": "kor_Hang",    # Korean
}

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using Meta's NLLB model."""
    global tokenizer, model

    if source_lang == target_lang:
        return text

    # Get NLLB language codes
    src_code = LANGUAGE_CODES.get(source_lang)
    tgt_code = LANGUAGE_CODES.get(target_lang)

    if not src_code or not tgt_code:
        raise ValueError(f"Unsupported language. Available: {list(LANGUAGE_CODES.keys())}")

    # Tokenize and translate
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cuda")
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
        max_length=512
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def handler(job):
    """RunPod serverless handler."""
    global tokenizer, model

    # Initialize model on first run (cold start)
    if tokenizer is None or model is None:
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

    try:
        payload = job["input"]
        text = payload.get("text", "")
        source_lang = payload.get("source_lang", "en")
        target_lang = payload.get("target_lang", "hi")

        if not text:
            return {"error": "No text provided"}

        translation = translate_text(text, source_lang, target_lang)
        return {
            "translation": translation,
            "source_lang": source_lang,
            "target_lang": target_lang
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})