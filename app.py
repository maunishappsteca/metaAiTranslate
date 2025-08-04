import os
from typing import Dict
from fairseq import checkpoint_utils, tasks
import torch
import runpod
from datetime import datetime

# Configuration
MODEL_PATH = "/app/models/nllb-200-distilled-600M.pt"  # Downloaded model
DATA_DIR = "/app/models"  # Contains dictionary files

# Initialize fairseq
def load_model():
    task = tasks.setup_task(task="translation", data=DATA_DIR)
    models, _ = checkpoint_utils.load_model_ensemble(
        [MODEL_PATH],
        task=task
    )
    model = models[0].to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model

model = load_model()

def translate(text: str, source_lang: str, target_lang: str) -> str:
    try:
        # Add language tags
        tagged_text = f"{source_lang} {target_lang} {text}"
        
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
    try:
        input = job["input"]
        text = input.get("text", "").strip()
        src_lang = input.get("source_lang", "").strip()
        tgt_lang = input.get("target_lang", "").strip()

        if not all([text, src_lang, tgt_lang]):
            return {"error": "Missing required fields"}

        translation = translate(text, src_lang, tgt_lang)
        return {
            "translation": translation,
            "source_lang": src_lang,
            "target_lang": tgt_lang
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})