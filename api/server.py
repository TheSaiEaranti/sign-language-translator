"""
api/server.py

FastAPI server with WebSocket endpoint for browser-based real-time inference.
Client streams base64-encoded frames; server returns predictions.

Run: uvicorn api.server:app --reload --port 8000
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image

from inference.engine import InferenceEngine


app = FastAPI(title="Sign Language Translator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton inference engine
LETTERS_CKPT = "checkpoints/letters_best.pt"
WORDS_CKPT = "checkpoints/words_best.pt"
engine = InferenceEngine(
    letters_ckpt=LETTERS_CKPT if Path(LETTERS_CKPT).exists() else None,
    words_ckpt=WORDS_CKPT if Path(WORDS_CKPT).exists() else None,
)


@app.get("/")
async def root():
    return {
        "service": "sign-language-translator",
        "endpoints": {
            "/ws/translate": "WebSocket endpoint for streaming inference",
            "/health": "Health check",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "letters_loaded": Path(LETTERS_CKPT).exists(), "words_loaded": Path(WORDS_CKPT).exists()}


@app.websocket("/ws/translate")
async def translate_ws(ws: WebSocket):
    """
    WebSocket endpoint. Client sends JSON: {"frame": "<base64-jpeg>"}
    Server replies: {"label": "A", "confidence": 0.92, "mode": "letter", "latency_ms": 28}
    """
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            payload = json.loads(msg)

            if "frame" not in payload:
                await ws.send_json({"error": "Missing 'frame' field"})
                continue

            try:
                img_bytes = base64.b64decode(payload["frame"].split(",")[-1])
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                frame_rgb = np.array(img)
            except Exception as e:
                await ws.send_json({"error": f"Invalid image: {e}"})
                continue

            pred = engine.predict(frame_rgb)
            if pred is None:
                await ws.send_json({"label": None, "confidence": 0.0, "mode": "none", "latency_ms": 0})
            else:
                await ws.send_json({
                    "label": pred.label,
                    "confidence": pred.confidence,
                    "mode": pred.mode,
                    "latency_ms": round(pred.latency_ms, 2),
                })

    except WebSocketDisconnect:
        pass


@app.post("/reset")
async def reset_engine():
    engine.reset()
    return {"status": "reset"}
