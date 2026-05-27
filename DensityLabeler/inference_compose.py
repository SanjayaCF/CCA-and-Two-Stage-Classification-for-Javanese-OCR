"""
inference_compose.py
====================
Javanese OCR — Full Inference + Unicode Composition Pipeline

Loads all trained models (Stage 1 + Stage 2) and processes a set of
structured crops produced by the segmentation pipeline, returning a
Javanese Unicode string per character group.

Usage
-----
# Process a directory of crops from one segmentation run:
    python inference_compose.py --crops-dir path/to/crops/

# Process a single pre-organised JSON manifest:
    python inference_compose.py --manifest path/to/manifest.json

# Demo mode (uses dummy data to verify the compose logic):
    python inference_compose.py --demo

Requirements: tensorflow, numpy, pillow
"""

import os
import sys
import json
import glob
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

# ─── Unicode tables (thesis §B.5) ────────────────────────────────────────────

HANACARAKA: dict[str, str] = {
    "ha" : "\uA9B2", "na" : "\uA9A4", "ca" : "\uA995", "ra" : "\uA9AB",
    "ka" : "\uA98F", "da" : "\uA9A0", "ta" : "\uA9A6", "sa" : "\uA9B1",
    "wa" : "\uA9B3", "la" : "\uA9AE", "pa" : "\uA9A5", "dha": "\uA99F",
    "ja" : "\uA997", "ya" : "\uA9B4", "nya": "\uA99E", "ma" : "\uA9A3",
    "ga" : "\uA98E", "ba" : "\uA9A7", "tha": "\uA9A1", "nga": "\uA98D",
}

ABOVE_MARKS: dict[str, str] = {
    "wulu"  : "\uA9B6",
    "pepet" : "\uA9BC",
    "cecak" : "\uA981",
    "layar" : "\uA982",
    "wigyan": "\uA983",
}

BELOW_MARKS: dict[str, str] = {
    "suku"  : "\uA9B8",
    "pengkal": "\uA9BE",
    "cakra" : "\uA9BD",
}

BESIDE_MARKS: dict[str, str] = {
    "pangkon": "\uA9C0",
    "tarung" : "\uA9B5",
}

WRAPPED_MARKS: dict[str, str] = {
    # taling (U+A9B4) before the base + tarung (U+A9B5) after
    "taling_tarung": ("\uA9B4", "\uA9B5"),
}

# ─── Unicode composer ─────────────────────────────────────────────────────────

def compose(
    base: str,
    above: str | None = None,
    below: str | None = None,
    beside: str | None = None,
    wrapped: str | None = None,
) -> str:
    """
    Compose individual position predictions into a Javanese Unicode string.

    Parameters
    ----------
    base    : one of HANACARAKA keys  (required)
    above   : one of ABOVE_MARKS keys, or None
    below   : one of BELOW_MARKS keys, or None
    beside  : one of BESIDE_MARKS keys, or None
    wrapped : one of WRAPPED_MARKS keys, or None

    Returns
    -------
    Unicode string in correct rendering order for Javanese script.
    """
    if base not in HANACARAKA:
        raise ValueError(f"Unknown base character: '{base}'. "
                         f"Valid: {list(HANACARAKA.keys())}")

    base_char = HANACARAKA[base]

    # Wrapped marks (taling-tarung) surround the base character
    if wrapped and wrapped in WRAPPED_MARKS:
        pre, post = WRAPPED_MARKS[wrapped]
        return pre + base_char + post

    # Normal composition order: base → above → below → beside
    result = base_char
    if above and above in ABOVE_MARKS:
        result += ABOVE_MARKS[above]
    if below and below in BELOW_MARKS:
        result += BELOW_MARKS[below]
    if beside and beside in BESIDE_MARKS:
        result += BESIDE_MARKS[beside]

    return result


def compose_from_dict(predictions: dict) -> str:
    """Convenience wrapper: accepts a dict with keys base/above/below/beside/wrapped."""
    return compose(
        base    = predictions.get("base"),
        above   = predictions.get("above"),
        below   = predictions.get("below"),
        beside  = predictions.get("beside"),
        wrapped = predictions.get("wrapped"),
    )


# ─── Image pre-processing ─────────────────────────────────────────────────────

def load_crop(path: str, img_size: int = 128) -> np.ndarray:
    """Load a grayscale crop, resize, normalise → shape (1, H, W, 1)."""
    img = Image.open(path).convert("L").resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, :, :, np.newaxis]   # (1, H, W, 1)


# ─── Two-Stage Classifier ─────────────────────────────────────────────────────

class TwoStageClassifier:
    """
    Loads all trained models and runs the full two-stage pipeline.

    models_dir  must contain:
        stage1_config.json          (+ .keras model)
        stage2_base_config.json     (+ .keras model)
        stage2_above_config.json    (optional)
        stage2_below_config.json    (optional)
        stage2_beside_config.json   (optional)
        stage2_wrapped_config.json  (optional)
    """

    POSITION_CONFIGS = {
        "base"   : "stage2_base_config.json",
        "above"  : "stage2_above_config.json",
        "below"  : "stage2_below_config.json",
        "beside" : "stage2_beside_config.json",
        "wrapped": "stage2_wrapped_config.json",
    }

    def __init__(self, models_dir: str = "models/"):
        import tensorflow as tf  # lazy import — not needed at module level
        self._tf = tf

        self.models_dir = models_dir
        self._stage1_model = None
        self._stage1_cfg   = None
        self._stage2_models: dict = {}
        self._stage2_cfgs: dict   = {}

        self._load_all()

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_json(self, filename: str) -> dict | None:
        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _load_keras_model(self, model_path: str):
        if not os.path.exists(model_path):
            return None
        return self._tf.keras.models.load_model(model_path, compile=False)

    def _load_all(self):
        print(f"Loading models from '{self.models_dir}' …")

        # Stage 1
        cfg = self._load_json("stage1_config.json")
        if cfg is None:
            raise FileNotFoundError(
                "stage1_config.json not found. Train Stage 1 first (Notebook 1).")
        model = self._load_keras_model(cfg["model_path"])
        if model is None:
            raise FileNotFoundError(f"Stage 1 model not found: {cfg['model_path']}")
        self._stage1_model = model
        self._stage1_cfg   = cfg
        print(f"  ✅ Stage 1  — binary classifier loaded  "
              f"(threshold={cfg['threshold']:.2f})")

        # Stage 2
        for pos, cfg_file in self.POSITION_CONFIGS.items():
            cfg = self._load_json(cfg_file)
            if cfg is None:
                print(f"  ⚠️  Stage 2 [{pos:<8}] — config not found, skipping")
                continue
            model = self._load_keras_model(cfg["model_path"])
            if model is None:
                print(f"  ⚠️  Stage 2 [{pos:<8}] — model not found, skipping")
                continue
            self._stage2_models[pos] = model
            self._stage2_cfgs[pos]   = cfg
            n_cls = len(cfg["class_indices"])
            print(f"  ✅ Stage 2 [{pos:<8}] — {n_cls} classes loaded")

        if "base" not in self._stage2_models:
            raise RuntimeError(
                "Base recogniser (stage2_base_config.json) is required for inference.")

    # ── Stage 1 ──────────────────────────────────────────────────────────────

    def is_valid(self, base_crop_path: str) -> tuple[bool, float]:
        """
        Returns (is_valid, confidence).
        is_valid=True  → pass to Stage 2.
        is_valid=False → discard as noise.
        """
        cfg = self._stage1_cfg
        arr = load_crop(base_crop_path, cfg["img_size"])
        prob = float(self._stage1_model.predict(arr, verbose=0)[0, 0])

        # Determine which index corresponds to "valid"
        valid_idx = cfg["class_indices"].get(cfg.get("valid_label", "valid"), 1)
        if valid_idx == 0:
            # valid is class 0 → model outputs P(class0); valid when prob < threshold
            is_valid = prob < cfg["threshold"]
            confidence = 1.0 - prob if is_valid else prob
        else:
            # valid is class 1 (default) → model outputs P(valid); valid when prob >= threshold
            is_valid = prob >= cfg["threshold"]
            confidence = prob if is_valid else 1.0 - prob

        return is_valid, confidence

    # ── Stage 2 ──────────────────────────────────────────────────────────────

    def classify_position(self, crop_path: str, position: str) -> tuple[str | None, float]:
        """
        Classify a single position crop.
        Returns (class_label, confidence) or (None, 0.0) if model unavailable.
        """
        if position not in self._stage2_models:
            return None, 0.0

        cfg   = self._stage2_cfgs[position]
        model = self._stage2_models[position]
        arr   = load_crop(crop_path, cfg["img_size"])
        probs = model.predict(arr, verbose=0)[0]

        # Invert class_indices map: int_index → label_str
        idx_to_label = {v: k for k, v in cfg["class_indices"].items()}
        best_idx     = int(np.argmax(probs))
        confidence   = float(probs[best_idx])

        # Special handling for wrapped: apply confidence threshold
        if position == "wrapped":
            threshold = cfg.get("confidence_threshold", 0.80)
            if confidence < threshold:
                return None, confidence

        return idx_to_label[best_idx], confidence

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def process_character_group(self, crops: dict[str, str | None]) -> dict:
        """
        Process one character group through the full two-stage pipeline.

        Parameters
        ----------
        crops : dict with optional keys base/above/below/beside/wrapped,
                values are file paths or None.

        Returns
        -------
        dict with keys:
            valid       : bool
            stage1_conf : float
            predictions : dict (position → label)
            confidences : dict (position → float)
            unicode     : str  (composed Javanese Unicode, or "" if invalid)
        """
        base_path = crops.get("base")
        if base_path is None or not os.path.exists(base_path):
            return {"valid": False, "stage1_conf": 0.0,
                    "predictions": {}, "confidences": {}, "unicode": ""}

        # Stage 1 — quality gate
        valid, s1_conf = self.is_valid(base_path)
        if not valid:
            return {"valid": False, "stage1_conf": s1_conf,
                    "predictions": {}, "confidences": {}, "unicode": ""}

        # Stage 2 — classify each available position
        predictions: dict[str, str | None] = {}
        confidences: dict[str, float]      = {}

        for pos in ("base", "above", "below", "beside", "wrapped"):
            path = crops.get(pos)
            if path and os.path.exists(path):
                label, conf = self.classify_position(path, pos)
                predictions[pos] = label
                confidences[pos] = conf
            else:
                predictions[pos] = None
                confidences[pos] = 0.0

        # Unicode composition
        try:
            unicode_char = compose_from_dict(predictions)
        except (ValueError, KeyError) as e:
            unicode_char = f"[ERR:{e}]"

        return {
            "valid"      : True,
            "stage1_conf": s1_conf,
            "predictions": predictions,
            "confidences": confidences,
            "unicode"    : unicode_char,
        }

    def process_crops_dir(self, crops_dir: str) -> list[dict]:
        """
        Auto-discover character groups from a flat crops directory produced by
        density_processing.process_image_density_guided().

        Expected filename pattern:
            density_{name}_{timestamp}_char_{N}_{position}.jpg

        Files for the same character index N are grouped together.
        Returns a list of result dicts, sorted by character index.
        """
        import re
        groups: dict[int, dict] = {}
        pattern = re.compile(r"_char_(\d+)_(base|above|below|beside|wrapped)\.jpg$")

        for path in glob.glob(os.path.join(crops_dir, "*.jpg")):
            m = pattern.search(os.path.basename(path))
            if not m:
                continue
            idx = int(m.group(1))
            pos = m.group(2)
            groups.setdefault(idx, {})[pos] = path

        results = []
        for idx in sorted(groups.keys()):
            result = self.process_character_group(groups[idx])
            result["char_index"] = idx
            results.append(result)

        return results

    def process_cropped_results(self, cropped_results: list[dict],
                                 output_dir: str = "") -> list[dict]:
        """
        Process the ``cropped_results`` list returned directly by
        ``process_image_density_guided()``.

        Parameters
        ----------
        cropped_results : the list of dicts with keys base/above/below/beside/wrapped.
                          Values are web-relative paths like 'static/results/...jpg'.
        output_dir      : optional prefix to convert web paths back to disk paths.
                          Leave empty if paths are already absolute or relative to cwd.

        Returns
        -------
        List of result dicts, one per character group.
        """
        results = []
        for idx, entry in enumerate(cropped_results):
            # Convert web-relative paths to disk paths when output_dir is given
            crops: dict[str, str | None] = {}
            for pos in ("base", "above", "below", "beside", "wrapped"):
                web_path = entry.get(pos)
                if web_path:
                    # Strip 'static/results/' prefix if present, prepend output_dir
                    rel = web_path.replace("static/results/", "").replace("\\", "/")
                    disk_path = os.path.join(output_dir, rel) if output_dir else rel
                    crops[pos] = disk_path
                else:
                    crops[pos] = None
            result = self.process_character_group(crops)
            result["char_index"] = idx
            results.append(result)
        return results


# ─── Pretty printing ──────────────────────────────────────────────────────────

def print_results(results: list[dict]) -> str:
    """Print a formatted table of results and return the composed line."""
    line = ""
    print("\n" + "═" * 70)
    print(f"  {'#':<4} {'Valid':<6} {'Base':<6} {'Above':<8} "
          f"{'Below':<8} {'Beside':<8} {'Wrapped':<12} {'Unicode'}")
    print("─" * 70)

    for r in results:
        idx  = r.get("char_index", "?")
        v    = "✅" if r["valid"] else "❌"
        p    = r.get("predictions", {})
        base    = (p.get("base")    or "—")[:6]
        above   = (p.get("above")   or "—")[:8]
        below   = (p.get("below")   or "—")[:8]
        beside  = (p.get("beside")  or "—")[:8]
        wrapped = (p.get("wrapped") or "—")[:12]
        uni     = r.get("unicode", "")
        line   += uni
        print(f"  {idx:<4} {v:<6} {base:<6} {above:<8} "
              f"{below:<8} {beside:<8} {wrapped:<12} {uni}")

    print("═" * 70)
    print(f"\n  Composed line : {line}")
    print(f"  Characters    : {len(results)} groups "
          f"({sum(1 for r in results if r['valid'])} valid)\n")
    return line


# ─── Demo mode ───────────────────────────────────────────────────────────────

def run_demo():
    """Test the compose() function with hard-coded examples (no models needed)."""
    print("\n" + "=" * 60)
    print("  DEMO — Unicode Composition (no models required)")
    print("=" * 60)

    test_cases = [
        dict(base="ha",  above=None,    below=None,   beside=None,      wrapped=None,          note="plain ha"),
        dict(base="na",  above="wulu",  below=None,   beside=None,      wrapped=None,          note="na + wulu (ni)"),
        dict(base="ka",  above="pepet", below="suku", beside=None,      wrapped=None,          note="ka + pepet + suku"),
        dict(base="ra",  above=None,    below=None,   beside="pangkon",  wrapped=None,         note="ra + pangkon (ra-mati)"),
        dict(base="ba",  above=None,    below=None,   beside=None,      wrapped="taling_tarung",note="ba + taling-tarung"),
        dict(base="ma",  above="cecak", below=None,   beside=None,      wrapped=None,          note="ma + cecak (mang)"),
        dict(base="sa",  above="layar", below="cakra",beside=None,      wrapped=None,          note="sa + layar + cakra"),
        dict(base="nga", above=None,    below="pengkal",beside=None,    wrapped=None,           note="nga + pengkal"),
    ]

    print(f"\n  {'Note':<30} {'Unicode':<12} {'Codepoints'}")
    print("  " + "─" * 60)
    for tc in test_cases:
        note = tc.pop("note")
        try:
            result = compose(**tc)
            cps    = " ".join(f"U+{ord(c):04X}" for c in result)
            print(f"  {note:<30} {result:<12} {cps}")
        except Exception as e:
            print(f"  {note:<30} ERROR: {e}")

    print()

    # Full line composition
    line_preds = [
        {"base": "ha",  "above": None,    "below": None,    "beside": None},
        {"base": "na",  "above": "wulu",  "below": None,    "beside": None},
        {"base": "ca",  "above": None,    "below": "suku",  "beside": None},
        {"base": "ra",  "above": None,    "below": None,    "beside": "pangkon"},
        {"base": "ka",  "above": "pepet", "below": None,    "beside": None},
    ]
    composed_line = "".join(compose_from_dict(p) for p in line_preds)
    print(f"  Composed line (demo) : {composed_line}")
    print(f"  Rendering hint       : font-family: 'Noto Sans Javanese', serif;")
    print()


# ─── CLI entry-point ──────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Javanese OCR — Two-Stage Inference + Unicode Composition",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--crops-dir", metavar="DIR",
        help="Flat directory of .jpg crops from density_processing.py\n"
             "(files named density_{name}_{ts}_char_{N}_{position}.jpg)")
    group.add_argument(
        "--cropped-results", metavar="FILE",
        help="JSON file containing the 'cropped_results' list returned by\n"
             "process_image_density_guided() — fastest integration path")
    group.add_argument(
        "--manifest", metavar="FILE",
        help="JSON file: list of {base, above, below, beside, wrapped} disk-path dicts")
    group.add_argument(
        "--demo", action="store_true",
        help="Run compose() demo (no models required)")
    p.add_argument(
        "--models-dir", metavar="DIR", default="models/",
        help="Directory containing trained model files (default: models/)")
    p.add_argument(
        "--output-dir", metavar="DIR", default="",
        help="Root directory of the Flask app (used to resolve web-relative paths\n"
             "in --cropped-results mode). E.g. the folder containing 'static/').")
    p.add_argument(
        "--output-json", metavar="FILE",
        help="Save results as JSON to this path")
    return p


def main():
    args = build_argparser().parse_args()

    if args.demo:
        run_demo()
        return

    classifier = TwoStageClassifier(models_dir=args.models_dir)

    if args.crops_dir:
        results = classifier.process_crops_dir(args.crops_dir)

    elif args.cropped_results:
        with open(args.cropped_results, encoding="utf-8") as f:
            data = json.load(f)
        # Accept either {"cropped_results": [...]} or a bare list
        cr = data.get("cropped_results", data) if isinstance(data, dict) else data
        results = classifier.process_cropped_results(
            cr, output_dir=args.output_dir or "")

    else:  # manifest
        with open(args.manifest) as f:
            manifest = json.load(f)
        results = [classifier.process_character_group(entry) for entry in manifest]
        for i, r in enumerate(results):
            r["char_index"] = i

    composed_line = print_results(results)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"line": composed_line, "characters": results}, f,
                      indent=2, ensure_ascii=False)
        print(f"Results saved → {args.output_json}")


if __name__ == "__main__":
    main()
