import os
from typing import Dict
from fairseq import checkpoint_utils, tasks
import torch
import runpod
from datetime import datetime

# Configuration
MODEL_PATH = "/app/models/nllb-200-distilled-600M.pt"
DATA_DIR = "/app/models"

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

def load_model():
    """Initialize fairseq model with proper task setup"""
    # Setup translation task
    task = tasks.setup_task(
        task="translation",
        data=DATA_DIR,
        source_lang="src",  # Placeholder
        target_lang="tgt"   # Placeholder
    )
    
    # Load model
    models, _ = checkpoint_utils.load_model_ensemble(
        [MODEL_PATH],
        task=task
    )
    model = models[0].to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model

model = load_model()

def translate(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text with proper language code handling"""
    try:
        # Get NLLB language codes
        src_code = LANGUAGE_CODES.get(source_lang)
        tgt_code = LANGUAGE_CODES.get(target_lang)
        
        if not src_code or not tgt_code:
            return f"Unsupported language pair: {source_lang}→{target_lang}"

        # Format as required by NLLB: [lang_code] text
        tagged_text = f"{src_code} {tgt_code} {text}"
        
        # Tokenize
        tokens = model.task.source_dictionary.encode_line(
            tagged_text,
            add_if_not_exist=False
        ).to(model.device)
        
        # Generate translation
        output = model.generate(
            tokens.unsqueeze(0),
            max_len=512
        )
        
        return model.task.target_dictionary.string(output[0]["tokens"])
    
    except Exception as e:
        return f"Translation failed: {str(e)}"

def handler(job):
    """RunPod handler with full validation"""
    try:
        input = job["input"]
        text = input.get("text", "").strip()
        src_lang = input.get("source_lang", "").strip().lower()
        tgt_lang = input.get("target_lang", "").strip().lower()

        # Validation
        if not text:
            return {"error": "Text cannot be empty"}
        if not src_lang:
            return {"error": "source_lang is required"}
        if not tgt_lang:
            return {"error": "target_lang is required"}
        if src_lang not in LANGUAGE_CODES:
            return {"error": f"Unsupported source language: {src_lang}"}
        if tgt_lang not in LANGUAGE_CODES:
            return {"error": f"Unsupported target language: {tgt_lang}"}

        # Perform translation
        translation = translate(text, src_lang, tgt_lang)
        if translation.startswith("Translation failed") or translation.startswith("Unsupported"):
            return {"error": translation}

        return {
            "translation": translation,
            "source_lang": src_lang,
            "target_lang": tgt_lang
        }
        
    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}

if __name__ == "__main__":
   runpod.serverless.start({"handler": handler})