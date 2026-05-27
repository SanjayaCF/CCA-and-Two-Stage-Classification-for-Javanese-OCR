"""
Stage 1 Inference — Standalone Script
Classifies every image in INPUT_FOLDER as valid aksara or noise.

Usage:
    python inference_stage1.py

Edit INPUT_FOLDER and MODEL_PATH below as needed.
"""

import os
import sys
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FOLDER = r"C:\Users\SanjayaCF\Desktop\Skripsi\labeling_tool_density\dataset_binary\noise"
MODEL_PATH   = None   # set to an explicit path to override auto-detect
THRESHOLD    = 0.48
IMG_SIZE     = 128
# ─────────────────────────────────────────────────────────────────────────────


def _find_model():
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "..", "OCR_Pipeline", "tahap1", "stage1_binary_classifier.keras"),
        os.path.join(base, "models", "stage1_binary_classifier.keras"),
    ]
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.exists(p):
            return p
    return None


def main():
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow not installed.  pip install tensorflow")
        sys.exit(1)

    model_path = MODEL_PATH or _find_model()
    if not model_path:
        print("ERROR: No model file found. Sync from Colab or set MODEL_PATH manually.")
        sys.exit(1)

    # Try loading threshold from saved config
    threshold = THRESHOLD
    for cfg_candidate in [
        model_path.replace(".keras", "_config.json"),
        os.path.join(os.path.dirname(model_path), "stage1_config.json"),
    ]:
        if os.path.exists(cfg_candidate):
            import json
            with open(cfg_candidate) as f:
                cfg = json.load(f)
            threshold = cfg.get("threshold", THRESHOLD)
            break

    print(f"Model    : {model_path}")
    print(f"Threshold: {threshold}")
    print(f"Folder   : {INPUT_FOLDER}")
    print()

    model = tf.keras.models.load_model(model_path)

    exts = ('.jpg', '.jpeg', '.png')
    images = sorted(f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(exts))
    if not images:
        print("No images found in INPUT_FOLDER.")
        sys.exit(0)

    print(f"{'#':<5} {'File':<55} {'Label':<8} {'P(valid)':>9}")
    print("-" * 82)

    n_valid = n_noise = 0
    for i, fname in enumerate(images):
        fpath = os.path.join(INPUT_FOLDER, fname)
        img = tf.keras.utils.load_img(fpath, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
        arr = tf.keras.utils.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        prob = float(model.predict(arr, verbose=0)[0][0])
        is_valid = prob >= threshold
        label = "valid" if is_valid else "noise"
        if is_valid:
            n_valid += 1
        else:
            n_noise += 1
        print(f"{i+1:<5} {fname[:54]:<55} {label:<8} {prob:>9.4f}")

    total = n_valid + n_noise
    print("-" * 82)
    print(f"Total: {total}  |  Valid: {n_valid} ({n_valid/total*100:.1f}%)  |  Noise: {n_noise} ({n_noise/total*100:.1f}%)")


if __name__ == "__main__":
    main()
