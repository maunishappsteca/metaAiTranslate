import torch
from fairseq.models.transformer import TransformerModel
import runpod

# ---------------------
# Load Model Once
# ---------------------
model = TransformerModel.from_pretrained(
    model_name_or_path='models/en-ru',
    checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
    data_name_or_path='models/en-ru',
    bpe='fastbpe',
    bpe_codes='models/en-ru/bpecodes'
).eval()

if torch.cuda.is_available():
    model = model.cuda()


# ---------------------
# RunPod Serverless Handler
# ---------------------
def handler(event):
    body = event.get("input", {}).get("body", {})
    text = body.get("text", "").strip()

    if not model:
        return {"error": "Model not loaded"}

    if not text:
        return {"error": "No input text provided"}

    try:
        translation = model.translate(text)
        return {"translation": translation}
    except Exception as e:
        return {"error": str(e)}


# ---------------------
# Start RunPod Handler
# ---------------------
runpod.serverless.start({"handler": handler})
