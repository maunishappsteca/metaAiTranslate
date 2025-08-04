import torch
from fairseq.models.transformer import TransformerModel
import runpod

# GPU Configuration
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for Ampere
torch.backends.cudnn.allow_tf32 = True

# Load model with AMP (Automatic Mixed Precision)
model = TransformerModel.from_pretrained(
    model_name_or_path='/app/models',
    checkpoint_file='nllb-200-distilled-600M.pt',
    data_name_or_path='/app/models',
    bpe='sentencepiece',
    fp16=True  # Enable mixed precision
).cuda()

def handler(job):
    try:
        input = job["input"]
        with torch.inference_mode():
            translation = model.translate(
                input["text"],
                src_lang=input["source_lang"],
                tgt_lang=input["target_lang"]
            )
        return {
            "translation": translation,
            "source_lang": input["source_lang"],
            "target_lang": input["target_lang"],
            "device": str(model.device)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print(f"Model loaded on {model.device}")
    runpod.serverless.start({"handler": handler})
    