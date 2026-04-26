# Real-Time Sign Language Translator

Real-time American Sign Language (ASL) translator with MediaPipe hand tracking and a custom hybrid architecture for letters (A–Z) and dynamic words/phrases.

## Architecture

```
data/         → Landmark extraction, augmentation, dataset loading
models/       → StaticGestureNet (letters) + DynamicGestureNet (words)
inference/    → Real-time webcam pipeline, <100ms latency target
training/     → Training loops, eval, checkpointing
api/          → FastAPI WebSocket server for browser clients
ui/           → React webcam UI
tests/        → Unit tests + benchmarks
```

## Two-model approach

**StaticGestureNet** — MLP over 21 hand landmarks (x, y, z) per frame. Classifies single-frame letters A–Z. Tiny (~50K params), runs in <2ms.

**DynamicGestureNet** — Transformer encoder over a temporal window of landmark sequences (30 frames @ 30fps = 1 second). Classifies word-level signs. ~500K params.

A lightweight **MotionDetector** routes each frame: if hands are static → letters model; if moving → buffer 30 frames → words model.

## Quickstart

```bash
pip install -r requirements.txt

# Train on letters
python -m training.train --task letters --epochs 50

# Train on words (uses WLASL or custom dataset)
python -m training.train --task words --epochs 100

# Run real-time webcam demo
python -m inference.webcam_demo

# Or launch the API + UI
uvicorn api.server:app --reload
cd ui && npm run dev
```
