"""
Aplikasi evaluasi OCR Aksara Jawa.
Jalankan setelah select_samples.py: python app.py
"""
import json, os, re, sys
from pathlib import Path
import importlib.util

from flask import Flask, jsonify, render_template, request

# ── Path setup ────────────────────────────────────────────────────────────────
_BASE    = Path(__file__).parent.resolve()
_SKRIPSI = _BASE.parent

# OCR_Pipeline processing modules (stage1, stage2, composition)
sys.path.insert(0, str(_SKRIPSI / "OCR_Pipeline"))

# density_processing_v2 dimuat langsung via path (sama seperti OCR_Pipeline/app.py)
_DP_PATH = _SKRIPSI / "Kerja Praktik" / "density_processing_v2.py"
_spec = importlib.util.spec_from_file_location("density_processing_v2", str(_DP_PATH))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
process_image_density_guided = _mod.process_image_density_guided

from processing import stage1, stage2, composition  # noqa: E402

# ── Flask & folder setup ──────────────────────────────────────────────────────
app = Flask(__name__)

TEST_IMAGES  = _BASE / "static" / "test_images"
CROPS_BASE   = _BASE / "static" / "crops"
CACHE_DIR    = _BASE / "cache"
RESULTS_FILE = _BASE / "eval_results.json"

for _d in [TEST_IMAGES, CROPS_BASE, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_PARAMS = {
    "threshold":                       160,
    "kernel_size":                     3,
    "min_area":                        15,
    "main_zone_ratio":                 0.55,
    "valley_ratio":                    0.22,
    "anchor_overlap_ratio":            0.35,
    "min_anchor_area_ratio":           0.50,
    "main_merge_gap":                  6,
    "attachment_horizontal_gap_ratio": 0.60,
    "attachment_vertical_gap_ratio":   0.75,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_image_list():
    return sorted(f.name for f in TEST_IMAGES.iterdir()
                  if f.suffix.lower() in {".jpg", ".png", ".jpeg"})

def load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return {}

def save_results_to_disk(data):
    RESULTS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def img_stem(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", Path(name).stem)

def crop_to_url(abs_path) -> str | None:
    """Absolute path di dalam static/ → URL /static/..."""
    if not abs_path:
        return None
    try:
        rel = Path(abs_path).resolve().relative_to((_BASE / "static").resolve())
        return "/static/" + str(rel).replace("\\", "/")
    except ValueError:
        return None

def normalize_crop_paths(seg, crops_dir: Path):
    """Pastikan semua path crop di cropped_results bersifat absolut."""
    for crop in seg.get("cropped_results", []):
        for pos in ("base", "above", "below", "beside", "wrapped"):
            p = crop.get(pos)
            if not p:
                continue
            abs_p = Path(p)
            if not abs_p.is_absolute():
                # Coba resolve relatif terhadap crops_dir
                candidate = crops_dir / abs_p.name
                if candidate.exists():
                    abs_p = candidate
                else:
                    abs_p = Path.cwd() / abs_p
            crop[pos] = str(abs_p.resolve())

# ── Pipeline runner (dengan cache) ────────────────────────────────────────────

def run_pipeline(img_name: str) -> dict | None:
    stem       = img_stem(img_name)
    cache_file = CACHE_DIR / f"{stem}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    crops_dir = CROPS_BASE / stem
    crops_dir.mkdir(exist_ok=True)

    img_path = str(TEST_IMAGES / img_name)
    seg = process_image_density_guided(img_path, str(crops_dir), **DEFAULT_PARAMS)
    if not seg:
        return None

    normalize_crop_paths(seg, crops_dir)

    for crop in seg.get("cropped_results", []):
        base_path       = crop.get("base")
        crop["stage1"]  = stage1.predict(base_path) if base_path else None
        s1              = crop.get("stage1")
        run_s2          = s1 is None or s1.get("valid", True)
        crop["stage2"]  = stage2.predict_crop(crop) if run_s2 else None
        crop["composition"] = (
            composition.compose_crop(crop["stage2"]) if crop["stage2"] else None
        )

    valid_comps = []
    for crop in seg.get("cropped_results", []):
        s1 = crop.get("stage1")
        if s1 is None or s1.get("valid", True):
            valid_comps.append(crop.get("composition"))

    final_u, final_l, _ = composition.compose_line(valid_comps)

    # Bangun data token untuk frontend
    tokens = []
    for crop in seg.get("cropped_results", []):
        s1 = crop.get("stage1")
        if s1 is not None and not s1.get("valid", True):
            continue
        comp = crop.get("composition") or {}
        s2   = crop.get("stage2") or {}

        def lbl(pos):
            p = s2.get(pos)
            return p["label"] if p else None

        def conf(pos):
            p = s2.get(pos)
            return p["confidence_pct"] if p else None

        tokens.append({
            "unicode": comp.get("unicode"),
            "latin":   comp.get("latin"),
            "labels":  {
                "base":   lbl("base"),
                "above":  lbl("above"),
                "below":  lbl("below"),
                "beside": lbl("beside"),
            },
            "confidence": {
                "base":   conf("base"),
                "above":  conf("above"),
                "below":  conf("below"),
                "beside": conf("beside"),
            },
            "crops": {
                "base":   crop_to_url(crop.get("base")),
                "above":  crop_to_url(crop.get("above")),
                "below":  crop_to_url(crop.get("below")),
                "beside": crop_to_url(crop.get("beside")),
            },
        })

    result = {"final_unicode": final_u, "final_latin": final_l, "tokens": tokens}
    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    images   = get_image_list()
    results  = load_results()
    reviewed = sum(1 for img in images if img in results)
    start    = next((i for i, img in enumerate(images) if img not in results), 0)
    return render_template(
        "eval.html",
        total=len(images),
        reviewed=reviewed,
        start_idx=start,
    )


@app.route("/api/image/<int:idx>")
def api_image(idx):
    images = get_image_list()
    if not 0 <= idx < len(images):
        return jsonify({"error": "Index di luar jangkauan"}), 404

    img_name = images[idx]
    results  = load_results()
    data     = run_pipeline(img_name)

    if not data:
        return jsonify({
            "image": img_name, "idx": idx, "total": len(images),
            "error": "Pipeline gagal — tidak ada komponen terdeteksi",
        })

    return jsonify({
        "image":   img_name,
        "idx":     idx,
        "total":   len(images),
        "reviewed": img_name in results,
        "review":  results.get(img_name),
        **data,
    })


@app.route("/api/save", methods=["POST"])
def api_save():
    body     = request.get_json() or {}
    img_name = body.get("image")
    review   = body.get("review")
    if not img_name:
        return jsonify({"error": "Missing image"}), 400
    results = load_results()
    results[img_name] = review
    save_results_to_disk(results)
    return jsonify({"ok": True})


@app.route("/api/summary")
def api_summary():
    images  = get_image_list()
    results = load_results()

    correct = wrong = seg_err = unknown = missing = 0
    for img in images:
        if img not in results:
            continue
        r = results[img]
        for t in r.get("tokens", []):
            st = t.get("status", "unreviewed")
            if   st == "correct":   correct  += 1
            elif st == "wrong":     wrong    += 1
            elif st == "seg_error": seg_err  += 1
            elif st == "unknown":   unknown  += 1
        missing += r.get("missing_count", 0)

    detected = correct + wrong + unknown
    gt_total = detected + missing

    return jsonify({
        "reviewed":               sum(1 for img in images if img in results),
        "total_images":           len(images),
        "correct":                correct,
        "wrong":                  wrong,
        "seg_errors":             seg_err,
        "unknown":                unknown,
        "missing":                missing,
        "gt_total":               gt_total,
        "segmentation_accuracy":  round(detected / gt_total * 100, 2) if gt_total else 0,
        "recognition_accuracy":   round(correct  / detected * 100, 2) if detected else 0,
        "end_to_end_accuracy":    round(correct  / gt_total * 100, 2) if gt_total else 0,
        "over_segmentation_rate": round(seg_err / (detected + seg_err) * 100, 2)
                                  if (detected + seg_err) else 0,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5003)
