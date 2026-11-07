import os
import tempfile
import base64
import io
import librosa
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt

# ==========================================
# 1. MODEL DEFINITION
# ==========================================
class AudioSpoofDetector(nn.Module):
    def __init__(self):
        super(AudioSpoofDetector, self).__init__()
        self.backbone = resnet18(weights=None) 
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)

# ==========================================
# 2. SETUP APP & GLOBALS
# ==========================================
app = FastAPI(title="Deepfake Audio Detector")

# Safely resolve the template directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Server is initializing model on device: {device}")

model = AudioSpoofDetector().to(device)

# CRITICAL FIX: Absolute path resolution
# This guarantees it finds the file regardless of where you launched Uvicorn
model_filename = 'best_audio_spoof_model_unweighted.pth' # <-- Updated Name!
model_path = os.path.join(BASE_DIR, model_filename)

if not os.path.exists(model_path):
    print(f"\nFATAL ERROR: '{model_filename}' not found at {model_path}!")
    print("Please make sure the file name is exact and it is in the same folder as app.py.")
else:
    try:
        # 1. Load the raw dictionary from the file into a variable
        state_dict = torch.load(model_path, map_location=device)

        # 2. Translate the keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('network.'):
                new_key = key.replace('network.', 'backbone.', 1)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
                
        # 3. Load the corrected state dict into your model with strict=False
        model.load_state_dict(new_state_dict, strict=False)

        print("Model weights loaded successfully! The AI is awake.")
    except Exception as e:
        print(f"WARNING: Failed to load weights. Error: {e}")

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

magma_cmap = plt.get_cmap('magma')

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def process_audio_to_tensor(file_path):
    target_seconds = 4
    sample_rate = 16000
    target_length = target_seconds * sample_rate
    
    y, sr = librosa.load(file_path, sr=sample_rate)
    y_fixed = librosa.util.fix_length(data=y, size=target_length)
    
    mel_spec = librosa.feature.melspectrogram(y=y_fixed, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    db_min, db_max = mel_spec_db.min(), mel_spec_db.max()
    if db_max != db_min:
        mel_normalized = (mel_spec_db - db_min) / (db_max - db_min)
    else:
        mel_normalized = np.zeros_like(mel_spec_db)
        
    magma_rgba = magma_cmap(mel_normalized)
    magma_rgba = np.flipud(magma_rgba)
    
    magma_rgb = (magma_rgba[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(magma_rgb)
    
    tensor = transform_pipeline(img).unsqueeze(0) 
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return tensor, img_base64

# ==========================================
# 4. API ENDPOINTS
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(('.flac', '.wav', '.mp3')):
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
            "spectrogram": base64_image
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)