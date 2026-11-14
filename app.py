import os
import tempfile
import base64
import io
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware

from spoof_detection import preprocess
from spoof_detection.model import CheckpointError, load_model_from_checkpoint

# ==========================================
# 1. SETUP APP & GLOBALS
# ==========================================
app = FastAPI(title="Deepfake Audio Detector")

# Safely resolve the template directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Server is initializing model on device: {device}")

# Absolute path resolution: finds the checkpoint regardless of where Uvicorn
# was launched.  The strict loader raises on a missing/mismatched state dict, so
# a broken remap can never leave part of the model randomly initialised
# (AGENTS.md G7).
model_filename = "best_audio_spoof_model_unweighted.pth"
model_path = os.path.join(BASE_DIR, model_filename)

try:
    model, model_load_report = load_model_from_checkpoint(model_path, device=device)
except (FileNotFoundError, CheckpointError) as exc:
    raise RuntimeError(
        f"FATAL: could not load '{model_filename}' at {model_path}: {exc}"
    ) from exc

print(
    "Model weights loaded successfully: "
    f"{model_load_report.state_dict_keys} keys, "
    f"{model_load_report.total_parameters} parameters, "
    f"missing={len(model_load_report.missing_keys)}, "
    f"unexpected={len(model_load_report.unexpected_keys)}."
)

transform_pipeline = preprocess.build_inference_transform()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def process_audio_to_tensor(file_path):
    """Apply the verified audio-to-image contract and return a model batch."""
    spec_db = preprocess.generate_mel_spectrogram(file_path)
    image = Image.fromarray(preprocess.render_mel_rgb(spec_db))

    tensor = transform_pipeline(image).unsqueeze(0)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return tensor, image_base64

# ==========================================
# 3. API ENDPOINTS
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".flac", ".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio.flush()
        temp_audio_path = temp_audio.name

    try:
        input_tensor, base64_image = process_audio_to_tensor(temp_audio_path)
        input_tensor = input_tensor.to(device)

        model.eval()

        with torch.no_grad():
            output = model(input_tensor).squeeze(1)
            probability = torch.sigmoid(output).item()

        is_spoof = probability >= 0.5
        label = "AI Generated (Spoof)" if is_spoof else "Real Human (Bonafide)"

        confidence = probability if is_spoof else (1 - probability)
        confidence_pct = round(confidence * 100, 2)

        return {
            "prediction": label,
            "confidence": f"{confidence_pct}%",
            "raw_probability": round(probability, 4),
            "spectrogram": base64_image,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)