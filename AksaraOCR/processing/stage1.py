"""
Stage 1 model wrapper — lazy-loads on first predict call.
Falls back gracefully if TensorFlow is not installed or model is missing.
"""

import os
import json
import numpy as np

_model     = None
_threshold = 0.48
_img_size  = 128
_loaded    = False
_error     = None


def _find_model():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base, "tahap1", "stage1_binary_classifier.keras"),
        os.path.join(base, "..", "Classification_Models", "models", "stage1_binary_classifier.keras"),
    ]
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.exists(p):
            return p
    return None


def _load():
    global _model, _threshold, _loaded, _error
    try:
        import tensorflow as tf
    except ImportError:
        _error = "TensorFlow not installed"
        return

    model_path = _find_model()
    if not model_path:
        _error = "Model file not found — sync from Colab first"
        return

    for cfg_path in [
        model_path.replace(".keras", "_config.json"),
        os.path.join(os.path.dirname(model_path), "stage1_config.json"),
    ]:
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            _threshold = cfg.get("threshold", _threshold)
            break

    try:
        _model  = tf.keras.models.load_model(model_path)
        _loaded = True
        print(f"[stage1] Loaded: {os.path.basename(model_path)}  threshold={_threshold}")
    except Exception as e:
        _error = str(e)


def predict(image_path):
    """
    Classify one base-crop image.
    Returns a dict or None if the model is unavailable.
    """
    global _loaded
    if not _loaded and _error is None:
        _load()
    if not _loaded:
        return None

    try:
        import tensorflow as tf
        img = tf.keras.utils.load_img(
            image_path, target_size=(_img_size, _img_size), color_mode="grayscale"
        )
        arr = tf.keras.utils.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        prob = float(_model.predict(arr, verbose=0)[0][0])
        is_valid = prob >= _threshold
        conf_pct = round((prob if is_valid else 1.0 - prob) * 100, 1)
        return {
            "valid":          is_valid,
            "prob":           round(prob, 4),
            "confidence_pct": conf_pct,
            "label":          "Valid" if is_valid else "Noise",
        }
    except Exception as e:
        print(f"[stage1] predict error: {e}")
        return None


def status():
    if not _loaded and _error is None:
        _load()
    if _loaded:
        return {"available": True, "threshold": _threshold}
    return {"available": False, "error": _error or "Unknown error"}
