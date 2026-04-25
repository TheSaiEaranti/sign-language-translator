"""
inference/webcam_demo.py

Real-time webcam demo. Press 'q' to quit, 'r' to reset buffer.

Usage:
    python -m inference.webcam_demo
    python -m inference.webcam_demo --letters-ckpt checkpoints/letters_best.pt
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from inference.engine import InferenceEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--letters-ckpt", type=str, default="checkpoints/letters_best.pt")
    parser.add_argument("--words-ckpt", type=str, default="checkpoints/words_best.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mirror", action="store_true", default=True)
    args = parser.parse_args()

    engine = InferenceEngine(
        letters_ckpt=args.letters_ckpt,
        words_ckpt=args.words_ckpt,
        device=args.device,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Real-time sign language translator")
    print("Press 'q' to quit, 'r' to reset buffer")

    fps_buffer = []
    last_label = ""

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            t0 = time.monotonic()
            pred = engine.predict(frame_rgb)
            frame_latency = (time.monotonic() - t0) * 1000

            fps_buffer.append(frame_latency)
            if len(fps_buffer) > 30:
                fps_buffer.pop(0)
            avg_latency = sum(fps_buffer) / len(fps_buffer)
            fps = 1000 / avg_latency if avg_latency > 0 else 0

            # Draw overlay
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (0, 0), (frame_bgr.shape[1], 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

            if pred and pred.confidence >= engine.confidence_threshold:
                last_label = f"{pred.label.upper()}  ({pred.mode})"
                conf_text = f"{pred.confidence:.2f}"
            else:
                conf_text = "—"

            cv2.putText(frame_bgr, last_label, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"conf: {conf_text}  |  {fps:.1f} fps  |  {avg_latency:.1f}ms",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("ASL Translator", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                engine.reset()
                last_label = ""

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
