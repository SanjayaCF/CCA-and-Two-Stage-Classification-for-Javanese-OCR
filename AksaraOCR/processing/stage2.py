"""
Stage 2 model wrappers — lazy-loads on first predict call.
Two models: base character classifier and unified sandhangan classifier.
Falls back gracefully if TensorFlow is not installed or models are missing.
"""

import os
import json
import numpy as np

_IMG_SIZE = 128

# Valid sandhangan classes per spatial position.
# Probabilities for classes outside the allowed set are zeroed before argmax.
POSITION_CLASSES = {
    "above":  {"wulu", "wulu_cecak", "pepet", "cecak", "layar",
               "wulu_layar", "pepet_cecak", "pepet_layar"},
    "below":  {"suku", "pengkal",
               "_ba", "_ca", "_da", "_dha", "_ja", "_ka",
               "_la", "_na", "_nga", "_ta", "_wa", "_ya"},
    "beside": {"pangkon"},
}

# Base classes that are atomic marks — no sandhangan can attach to them.
# Any crops detected around these are segmentation artifacts and should be ignored.
NO_SANDH_BASES = {
    "taling", "tarung", "wignyan",
    "pada_lingsa", "pada_lungsi",
    "_ha", "_pa", "_sa",
}

_base_model       = None
_base_classes     = []   # index -> label
_base_loaded      = False
_base_error       = None

_sandh_model      = None
_sandh_classes    = []
_sandh_loaded     = False
_sandh_error      = None


def _tahap2_dir():
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tahap2")
    )


def _load_one(model_name, config_name):
    """Load a single keras model + config. Returns (model, classes, error)."""
    try:
        import tensorflow as tf
    except ImportError:
        return None, [], "TensorFlow not installed"

    d = _tahap2_dir()
    model_path  = os.path.join(d, model_name)
    config_path = os.path.join(d, config_name)

    if not os.path.exists(model_path):
        return None, [], f"Model not found: {model_name} — run notebook2_stage2_classifier.ipynb first"
    if not os.path.exists(config_path):
        return None, [], f"Config not found: {config_name}"

    with open(config_path) as f:
        cfg = json.load(f)

    # class_indices maps label -> int index; invert to index -> label
    class_indices = cfg.get("class_indices", {})
    classes = [""] * len(class_indices)
    for label, idx in class_indices.items():
        classes[idx] = label

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"[stage2] Loaded {model_name}  classes={len(classes)}")
        return model, classes, None
    except Exception as e:
        return None, [], str(e)


def _load_base():
    global _base_model, _base_classes, _base_loaded, _base_error
    _base_model, _base_classes, _base_error = _load_one(
        "stage2_base.keras", "stage2_base_config.json"
    )
    _base_loaded = _base_model is not None


def _load_sandh():
    global _sandh_model, _sandh_classes, _sandh_loaded, _sandh_error
    _sandh_model, _sandh_classes, _sandh_error = _load_one(
        "stage2_sandhangan.keras", "stage2_sandhangan_config.json"
    )
    _sandh_loaded = _sandh_model is not None


def _predict_one(model, classes, image_path, allowed_labels=None):
    """Run inference on a single crop image. Returns {label, confidence_pct, probs}.

    If allowed_labels is given, probabilities for classes outside that set are
    zeroed before argmax so the result is always a position-valid class.
    """
    try:
        import tensorflow as tf
        img = tf.keras.utils.load_img(
            image_path, target_size=(_IMG_SIZE, _IMG_SIZE), color_mode="grayscale"
        )
        arr = tf.keras.utils.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        pv  = model.predict(arr, verbose=0)[0]

        if allowed_labels:
            masked = np.array([
                pv[i] if classes[i] in allowed_labels else 0.0
                for i in range(len(classes))
            ])
            idx = int(np.argmax(masked))
        else:
            idx = int(np.argmax(pv))

        conf = round(float(pv[idx]) * 100, 1)
        return {
            "label":          classes[idx] if idx < len(classes) else "unknown",
            "confidence_pct": conf,
            "probs":          {classes[i]: round(float(pv[i]), 4)
                               for i in range(len(classes))},
        }
    except Exception as e:
        print(f"[stage2] predict error: {e}")
        return None


def predict_crop(crop):
    """
    Classify all components in one segmentation crop dict.

    crop keys: base, above, below, beside, wrapped (file paths or None).

    Returns a dict with the same keys containing prediction results,
    or None for positions with no crop image.
    """
    global _base_loaded, _sandh_loaded

    if not _base_loaded and _base_error is None:
        _load_base()
    if not _sandh_loaded and _sandh_error is None:
        _load_sandh()

    result = {}

    base_path = crop.get("base")
    if base_path and os.path.exists(base_path) and _base_loaded:
        result["base"] = _predict_one(_base_model, _base_classes, base_path)

    # Atomic marks cannot carry sandhangan — skip surrounding positions entirely.
    base_label = (result["base"]["label"] if result.get("base") else None)
    if base_label in NO_SANDH_BASES:
        return result

    for pos in ("above", "below", "beside", "wrapped"):
        path = crop.get(pos)
        if path and os.path.exists(path) and _sandh_loaded:
            allowed = POSITION_CLASSES.get(pos)
            result[pos] = _predict_one(_sandh_model, _sandh_classes, path,
                                       allowed_labels=allowed)

    return result if result else None


def status():
    if not _base_loaded and _base_error is None:
        _load_base()
    if not _sandh_loaded and _sandh_error is None:
        _load_sandh()
    return {
        "base": {
            "available": _base_loaded,
            "classes":   len(_base_classes),
            "error":     _base_error,
        },
        "sandhangan": {
            "available": _sandh_loaded,
            "classes":   len(_sandh_classes),
            "error":     _sandh_error,
        },
    }
