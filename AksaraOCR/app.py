import os
import importlib.util

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Load density_processing_v2 directly by file path so it never conflicts
# with the local processing/ package.
_DP_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Kerja Praktik", "density_processing_v2.py"
))
_spec = importlib.util.spec_from_file_location("density_processing_v2", _DP_PATH)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
process_image_density_guided = _mod.process_image_density_guided

from processing import stage1, stage2, composition

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER = os.path.join("static", "uploads")
RESULT_FOLDER = os.path.join("static", "results")
ALLOWED_EXT   = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

_last_upload = None

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


def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _run(input_path, params):
    seg = process_image_density_guided(input_path, RESULT_FOLDER, **params)
    if not seg:
        return None

    s1_status = stage1.status()
    s2_status = stage2.status()

    for crop in seg.get("cropped_results", []):
        base_path = crop.get("base")
        crop["stage1"] = stage1.predict(base_path) if base_path else None
        # Only run Stage 2 for crops that Stage 1 marks as valid (or when unavailable)
        s1_result = crop.get("stage1")
        run_s2 = s1_result is None or s1_result.get("valid", True)
        crop["stage2"] = stage2.predict_crop(crop) if run_s2 else None
        crop["composition"] = composition.compose_crop(crop["stage2"]) if crop["stage2"] else None

    seg["stage1_available"]    = s1_status["available"]
    seg["stage1_error"]        = s1_status.get("error")
    seg["stage2_status"]       = s2_status

    # Collect valid-crop compositions (Stage 1 passed or unavailable)
    valid_comps = []
    for crop in seg.get("cropped_results", []):
        s1 = crop.get("stage1")
        if s1 is None or s1.get("valid", True):
            valid_comps.append(crop.get("composition"))

    # Raw text: per-character concatenation before cross-character rules
    seg["raw_unicode"] = "".join((c.get("unicode") or "") for c in valid_comps if c)
    seg["raw_latin"]   = "-".join((c.get("latin")   or "") for c in valid_comps if c and c.get("latin"))

    # Final text: after cross-character rules (taling, tarung, wignyan, pasangan)
    final_u, final_l, final_tokens = composition.compose_line(valid_comps)
    seg["final_unicode"]        = final_u
    seg["final_latin"]          = final_l
    seg["final_unicode_tokens"] = final_tokens

    return seg


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    global _last_upload
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400
    f = request.files["file"]
    if not f.filename or not _allowed(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename    = secure_filename(f.filename)
    input_path  = os.path.join(UPLOAD_FOLDER, filename)
    f.save(input_path)
    _last_upload = input_path

    params = {k: request.form.get(k, DEFAULT_PARAMS[k], type=type(DEFAULT_PARAMS[k]))
              for k in DEFAULT_PARAMS}
    result = _run(input_path, params)
    if not result:
        return jsonify({"error": "Segmentation failed or no components found"}), 500
    return jsonify(result)


@app.route("/reprocess", methods=["POST"])
def reprocess():
    if not _last_upload or not os.path.exists(_last_upload):
        return jsonify({"error": "No image uploaded yet"}), 400
    data   = request.get_json() or {}
    params = {k: type(DEFAULT_PARAMS[k])(data.get(k, DEFAULT_PARAMS[k])) for k in DEFAULT_PARAMS}
    result = _run(_last_upload, params)
    if not result:
        return jsonify({"error": "Segmentation failed"}), 500
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5002)
