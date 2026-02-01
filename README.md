🎙️ AI Generated Voice Detection API
Overview

This project provides a REST API that detects whether a voice sample is:

AI_GENERATED (synthetic speech)
or
HUMAN (real human voice)

It also identifies the spoken language and returns a calibrated confidence score with an explanation.

Supported languages:
English, Hindi, Tamil, Telugu, Malayalam

🚀 Features

AI vs Human voice classification

Confidence calibration to prevent overconfidence

Multi-language detection using speech recognition

Base64 audio input (MP3/WAV)

Explainable predictions

🧠 System Architecture

Client Audio (Base64)
        ↓
 FastAPI Endpoint
        ↓
 Audio Preprocessing (Mel Spectrogram)
        ↓
 CNN Detector (AI vs Human)
        ↓
 Whisper Language Detection
        ↓
 JSON Response

 📦 Technology Stack

| Component          | Tool              |
| ------------------ | ----------------- |
| Detection Model    | PyTorch CNN       |
| Audio Processing   | Librosa           |
| Language Detection | OpenAI Whisper    |
| API                | FastAPI + Uvicorn |

📥 Input Format

Base64 encoded MP3 or WAV audio.
Minimum recommended duration: ~3 seconds.

📤 Output Format

{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and synthetic spectral patterns detected"
}

🎯 Classification Rules

| Value        | Meaning                        |
| ------------ | ------------------------------ |
| AI_GENERATED | Synthetic / AI-produced speech |
| HUMAN        | Real human voice               |

📊 Model Approach

AI Voice Detection
A convolutional neural network trained on:
• Human speech (Common Voice, LibriSpeech)
• AI generated synthetic voices
Audio is converted to mel-spectrograms for robust pattern recognition.

Language Detection

Whisper speech recognition model automatically identifies the spoken language.

🔌 API Endpoint

POST /detect
Headers:
X-API-Key: voice_detector_2026

🏁 Summary

This system delivers:
Reliable AI voice detection
Language identification
Confidence-based decisions
Production-ready REST API
Built using ethical, transparent machine learning.

📈 Ready for real-world deployment.