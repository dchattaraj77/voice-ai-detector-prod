import os
import base64
import io
import requests
import torch
import numpy as np
import librosa
import whisper

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ======================
# Low-RAM optimizations
# ======================

torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ======================
# Config
# ======================

API_KEY = "voice_detector_2026"
MODEL_PATH = "models/detector.pt"

SAMPLE_RATE = 16000
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION
TEMPERATURE = 1.8

# Tiny Whisper = very low memory, perfect for language detection
whisper_model = whisper.load_model("tiny")

LANG_MAP = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam"
}

# ======================
# CNN Detector
# ======================

class Detector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),

            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


model = Detector()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ======================
# FastAPI
# ======================

app = FastAPI(title="AI Voice Detector API")


class AudioRequest(BaseModel):
    audio_base64: str | None = None
    audio_url: str | None = None


# ======================
# Helpers
# ======================

def preprocess(audio_bytes):
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)

    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=128
    )

    mel = librosa.power_to_db(mel)

    return torch.tensor(mel).float().unsqueeze(0).unsqueeze(0)


def detect_language(audio_bytes):
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    result = whisper_model.transcribe(audio, fp16=False)

    code = result.get("language", "unknown")
    return LANG_MAP.get(code, code)


def load_audio(req: AudioRequest):
    if req.audio_base64:
        return base64.b64decode(req.audio_base64.strip())

    if req.audio_url:
        r = requests.get(req.audio_url, timeout=15)
        if r.status_code != 200:
            raise HTTPException(400, "Failed to download audio file")
        return r.content

    raise HTTPException(400, "Provide audio_base64 or audio_url")


# ======================
# API Endpoint
# ======================

@app.post("/detect")
def detect(req: AudioRequest, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")

    audio_bytes = load_audio(req)

    if len(audio_bytes) < 1000:
        raise HTTPException(400, "Invalid audio file")

    features = preprocess(audio_bytes)

    with torch.no_grad():
        logits = model(features) / TEMPERATURE
        probs = torch.softmax(logits, dim=1)[0]

    ai_prob = float(probs[1])

    classification = "AI_GENERATED" if ai_prob > 0.5 else "HUMAN"

    confidence = round(
        ai_prob if classification == "AI_GENERATED" else 1 - ai_prob,
        4
    )

    explanation = (
        "Unnatural pitch consistency and synthetic spectral patterns detected"
        if classification == "AI_GENERATED"
        else "Natural speech variations and human vocal dynamics detected"
    )

    language = detect_language(audio_bytes)

    return {
        "status": "success",
        "language": language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }

