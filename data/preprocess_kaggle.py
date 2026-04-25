"""Kaggle ASL preprocessing — extract MediaPipe landmarks from images."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from data.landmarks import LandmarkExtractor

VALID_EXTS = {".jpg", ".jpeg", ".png"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", default="landmarks_data")
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist"); sys.exit(1)

    extractor = LandmarkExtractor(static_image_mode=True, min_detection_confidence=0.4)
    letter_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and len(d.name) == 1 and d.name.isalpha()])
    if not letter_dirs:
        print(f"No letter directories in {raw_dir}"); sys.exit(1)

    print(f"Found {len(letter_dirs)} letter directories\nProcessing up to {args.max_per_class} per letter\nOutput: {out_dir}\n")
    total_processed, total_skipped, total_no_hand = 0, 0, 0

    for letter_dir in letter_dirs:
        letter = letter_dir.name.upper()
        out_letter_dir = out_dir / letter
        out_letter_dir.mkdir(parents=True, exist_ok=True)
        images = sorted([p for p in letter_dir.iterdir() if p.suffix.lower() in VALID_EXTS])[: args.max_per_class]
        per_processed, per_no_hand = 0, 0

        for img_path in tqdm(images, desc=f"  {letter}", leave=False):
            out_path = out_letter_dir / (img_path.stem + ".npy")
            if args.skip_existing and out_path.exists():
                total_skipped += 1; continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            hand_frame = extractor.extract(img_rgb)
            if not hand_frame.has_hands():
                per_no_hand += 1; total_no_hand += 1; continue
            np.save(out_path, hand_frame.to_tensor())
            per_processed += 1; total_processed += 1

        print(f"{letter}: {per_processed} saved, {per_no_hand} no-hand")

    extractor.close()
    print(f"\n{'='*50}\nTotal processed: {total_processed}\nSkipped: {total_skipped}\nNo hand: {total_no_hand}\nOutput: {out_dir.absolute()}")
    print(f"\nNow train with:\n  python -m training.train --task letters --data-root {out_dir} --epochs 50")

if __name__ == "__main__":
    main()
