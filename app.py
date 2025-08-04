import os
import json
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import runpod
import pycld2 as cld2  # New import for language detection

# Initialize models (loaded once on cold start)
tokenizer = None
model = None

# Language code mapping (ISO 639-1 to NLLB codes)
LANGUAGE_CODES = {
    # A
    "aa": "aar_Latn",  # Afar
    "ab": "abk_Cyrl",  # Abkhazian
    "af": "afr_Latn",  # Afrikaans
    "ak": "aka_Latn",  # Akan
    "am": "amh_Ethi",  # Amharic
    "ar": "arb_Arab",  # Arabic (Standard)
    "an": "arg_Latn",  # Aragonese
    "as": "asm_Beng",  # Assamese
    "av": "ava_Cyrl",  # Avaric
    "ay": "aym_Latn",  # Aymara
    "az": "azj_Latn",  # Azerbaijani
    
    # B
    "ba": "bak_Cyrl",  # Bashkir
    "be": "bel_Cyrl",  # Belarusian
    "bg": "bul_Cyrl",  # Bulgarian
    "bh": "bho_Deva",  # Bhojpuri
    "bi": "bis_Latn",  # Bislama
    "bm": "bam_Latn",  # Bambara
    "bn": "ben_Beng",  # Bengali
    "bo": "bod_Tibt",  # Tibetan
    "br": "bre_Latn",  # Breton
    "bs": "bos_Latn",  # Bosnian
    
    # C
    "ca": "cat_Latn",  # Catalan
    "ce": "che_Cyrl",  # Chechen
    "ch": "cha_Latn",  # Chamorro
    "co": "cos_Latn",  # Corsican
    "cr": "cre_Latn",  # Cree
    "cs": "ces_Latn",  # Czech
    "cv": "chv_Cyrl",  # Chuvash
    "cy": "cym_Latn",  # Welsh
    
    # D
    "da": "dan_Latn",  # Danish
    "de": "deu_Latn",  # German
    "dv": "div_Thaa",  # Dhivehi
    "dz": "dzo_Tibt",  # Dzongkha
    
    # E
    "ee": "ewe_Latn",  # Ewe
    "el": "ell_Grek",  # Greek
    "en": "eng_Latn",  # English
    "eo": "epo_Latn",  # Esperanto
    "es": "spa_Latn",  # Spanish
    "et": "est_Latn",  # Estonian
    "eu": "eus_Latn",  # Basque
    
    # F
    "fa": "pes_Arab",  # Persian
    "ff": "ful_Latn",  # Fulah
    "fi": "fin_Latn",  # Finnish
    "fj": "fij_Latn",  # Fijian
    "fo": "fao_Latn",  # Faroese
    "fr": "fra_Latn",  # French
    "fy": "fry_Latn",  # Western Frisian
    
    # G
    "ga": "gle_Latn",  # Irish
    "gd": "gla_Latn",  # Scottish Gaelic
    "gl": "glg_Latn",  # Galician
    "gn": "grn_Latn",  # Guarani
    "gu": "guj_Gujr",  # Gujarati
    "gv": "glv_Latn",  # Manx
    
    # H
    "ha": "hau_Latn",  # Hausa
    "he": "heb_Hebr",  # Hebrew
    "hi": "hin_Deva",  # Hindi
    "ho": "hmo_Latn",  # Hiri Motu
    "hr": "hrv_Latn",  # Croatian
    "ht": "hat_Latn",  # Haitian
    "hu": "hun_Latn",  # Hungarian
    "hy": "hye_Armn",  # Armenian
    
    # I
    "ia": "ina_Latn",  # Interlingua
    "id": "ind_Latn",  # Indonesian
    "ie": "ile_Latn",  # Interlingue
    "ig": "ibo_Latn",  # Igbo
    "ii": "iii_Latn",  # Sichuan Yi
    "ik": "ipk_Latn",  # Inupiaq
    "io": "ido_Latn",  # Ido
    "is": "isl_Latn",  # Icelandic
    "it": "ita_Latn",  # Italian
    "iu": "iku_Latn",  # Inuktitut
    
    # J
    "ja": "jpn_Jpan",  # Japanese
    "jv": "jav_Latn",  # Javanese
    
    # K
    "ka": "kat_Geor",  # Georgian
    "kg": "kon_Latn",  # Kongo
    "ki": "kik_Latn",  # Kikuyu
    "kj": "kua_Latn",  # Kuanyama
    "kk": "kaz_Cyrl",  # Kazakh
    "kl": "kal_Latn",  # Kalaallisut
    "km": "khm_Khmr",  # Khmer
    "kn": "kan_Knda",  # Kannada
    "ko": "kor_Hang",  # Korean
    "kr": "kau_Latn",  # Kanuri
    "ks": "kas_Arab",  # Kashmiri
    "ku": "kmr_Latn",  # Kurdish
    "kv": "kom_Cyrl",  # Komi
    "kw": "cor_Latn",  # Cornish
    "ky": "kir_Cyrl",  # Kyrgyz
    
    # L
    "la": "lat_Latn",  # Latin
    "lb": "ltz_Latn",  # Luxembourgish
    "lg": "lug_Latn",  # Ganda
    "li": "lim_Latn",  # Limburgish
    "ln": "lin_Latn",  # Lingala
    "lo": "lao_Laoo",  # Lao
    "lt": "lit_Latn",  # Lithuanian
    "lu": "lua_Latn",  # Luba-Katanga
    "lv": "lvs_Latn",  # Latvian
    
    # M
    "mg": "mlg_Latn",  # Malagasy
    "mh": "mah_Latn",  # Marshallese
    "mi": "mri_Latn",  # Maori
    "mk": "mkd_Cyrl",  # Macedonian
    "ml": "mal_Mlym",  # Malayalam
    "mn": "khk_Cyrl",  # Mongolian (Khalkha)
    "mr": "mar_Deva",  # Marathi
    "ms": "zsm_Latn",  # Malay
    "mt": "mlt_Latn",  # Maltese
    "my": "mya_Mymr",  # Burmese
    
    # N
    "na": "nau_Latn",  # Nauru
    "nb": "nob_Latn",  # Norwegian Bokmål
    "nd": "nde_Latn",  # North Ndebele
    "ne": "npi_Deva",  # Nepali
    "ng": "ndo_Latn",  # Ndonga
    "nl": "nld_Latn",  # Dutch
    "nn": "nno_Latn",  # Norwegian Nynorsk
    "no": "nor_Latn",  # Norwegian
    "nr": "nbl_Latn",  # South Ndebele
    "nv": "nav_Latn",  # Navajo
    "ny": "nya_Latn",  # Nyanja
    
    # O
    "oc": "oci_Latn",  # Occitan
    "oj": "oji_Latn",  # Ojibwe
    "om": "orm_Latn",  # Oromo
    "or": "ory_Orya",  # Odia
    "os": "oss_Cyrl",  # Ossetian
    
    # P
    "pa": "pan_Guru",  # Punjabi
    "pi": "pli_Deva",  # Pali
    "pl": "pol_Latn",  # Polish
    "ps": "pbt_Arab",  # Pashto
    "pt": "por_Latn",  # Portuguese
    
    # Q
    "qu": "que_Latn",  # Quechua
    
    # R
    "rm": "roh_Latn",  # Romansh
    "rn": "run_Latn",  # Rundi
    "ro": "ron_Latn",  # Romanian
    "ru": "rus_Cyrl",  # Russian
    "rw": "kin_Latn",  # Kinyarwanda
    
    # S
    "sa": "san_Deva",  # Sanskrit
    "sc": "srd_Latn",  # Sardinian
    "sd": "snd_Arab",  # Sindhi
    "se": "sme_Latn",  # Northern Sami
    "sg": "sag_Latn",  # Sango
    "si": "sin_Sinh",  # Sinhala
    "sk": "slk_Latn",  # Slovak
    "sl": "slv_Latn",  # Slovenian
    "sm": "smo_Latn",  # Samoan
    "sn": "sna_Latn",  # Shona
    "so": "som_Latn",  # Somali
    "sq": "sqi_Latn",  # Albanian
    "sr": "srp_Cyrl",  # Serbian
    "ss": "ssw_Latn",  # Swati
    "st": "sot_Latn",  # Southern Sotho
    "su": "sun_Latn",  # Sundanese
    "sv": "swe_Latn",  # Swedish
    "sw": "swh_Latn",  # Swahili
    
    # T
    "ta": "tam_Taml",  # Tamil
    "te": "tel_Telu",  # Telugu
    "tg": "tgk_Cyrl",  # Tajik
    "th": "tha_Thai",  # Thai
    "ti": "tir_Ethi",  # Tigrinya
    "tk": "tuk_Latn",  # Turkmen
    "tl": "tgl_Latn",  # Tagalog
    "tn": "tsn_Latn",  # Tswana
    "to": "ton_Latn",  # Tonga
    "tr": "tur_Latn",  # Turkish
    "ts": "tso_Latn",  # Tsonga
    "tt": "tat_Cyrl",  # Tatar
    "tw": "twi_Latn",  # Twi
    "ty": "tah_Latn",  # Tahitian
    
    # U
    "ug": "uig_Arab",  # Uyghur
    "uk": "ukr_Cyrl",  # Ukrainian
    "ur": "urd_Arab",  # Urdu
    "uz": "uzn_Latn",  # Uzbek
    
    # V
    "ve": "ven_Latn",  # Venda
    "vi": "vie_Latn",  # Vietnamese
    "vo": "vol_Latn",  # Volapük
    
    # W
    "wa": "wln_Latn",  # Walloon
    "wo": "wol_Latn",  # Wolof
    
    # X
    "xh": "xho_Latn",  # Xhosa
    
    # Y
    "yi": "ydd_Hebr",  # Yiddish
    "yo": "yor_Latn",  # Yoruba
    
    # Z
    "za": "zha_Latn",  # Zhuang
    "zh": "zho_Hans",  # Chinese (Simplified)
    "zu": "zul_Latn"   # Zulu
}

def detect_language(text: str) -> str:
    """Auto-detect language using Compact Language Detector 2."""
    try:
        _, _, _, detected_lang = cld2.detect(text, bestEffort=True)
        lang_code = detected_lang[0].lower()  # Convert to lowercase ISO 639-1
        
        # Validate detected language is supported
        if lang_code not in LANGUAGE_CODES:
            print(f"Detected unsupported language: {lang_code}, defaulting to 'en'")
            return "en"
            
        return lang_code
    except Exception as e:
        print(f"Language detection failed: {str(e)}")
        return "en"  # Fallback to English

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using Meta's NLLB model."""
    global tokenizer, model

    if source_lang == "-":
        source_lang = detect_language(text)
        print(f"Auto-detected language: {source_lang}")

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

    try:
        # Initialize models if not loaded
        if tokenizer is None or model is None:
            model_name = "facebook/nllb-200-distilled-600M"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

        payload = job["input"]
        text = payload.get("text", "")
        source_lang = payload.get("source_lang", "en")
        target_lang = payload.get("target_lang", "hi")

        if not text:
            return {"error": "No text provided"}

        # Auto-detect if source_lang is "-"
        if source_lang == "-":
            source_lang = detect_language(text)
            print(f"Auto-detected source language: {source_lang}")

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