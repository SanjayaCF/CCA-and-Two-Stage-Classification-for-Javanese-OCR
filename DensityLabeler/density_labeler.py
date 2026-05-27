"""
density_labeler.py
==================
Web-based labeling tool for the Javanese OCR two-stage pipeline.
Uses density_processing.process_image_density_guided() for segmentation.

Modes
-----
  stage1  — label BASE crops as valid / noise
            Output: dataset_binary/valid/   and   dataset_binary/noise/

  stage2  — label ALL position crops with their specific class
            Output: dataset_stage2/{position}/{class}/

Usage
-----
  python density_labeler.py --input line_images/ --mode stage1
  python density_labeler.py --input line_images/ --mode stage2
  python density_labeler.py --input line_images/ --mode stage2 --port 5006
"""

import os
import sys
import shutil
import base64
import argparse

import cv2
from flask import Flask, jsonify, request, render_template, send_from_directory

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)  # parent of labeling_tool_density
kp_dir = os.path.join(root_dir, "Kerja Praktik")

sys.path.insert(0, kp_dir)

from density_processing_v2 import process_image_density_guided

# ── App & globals ─────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")

INPUT_DIR   = ""
DATASET_DIR = ""
CROPS_TMP   = ""
MODE        = "stage1"
ALL_FILES   = []

_seg_cache: dict = {}  # fname → process_image_density_guided() result (in-memory only)
REVIEW_ITEMS: list = []  # [{bucket, disk_path, fname}, ...] built at startup for review mode
CUSTOM_CLASSES_FILE = ""  # path to custom_classes.json; set in main()
REVIEW_FOLDER = ""        # "bucket/class" to review a specific labeled folder; empty = unlabeled

# ── Class / label definitions ─────────────────────────────────────────────────
STAGE1_LABELS = ["valid", "noise"]

# Training buckets: base (27 classes) and sandhangan (18 marks).
POSITION_CLASSES: dict[str, list[str]] = {
    "base": [
        "ha","na","ca","ra","ka","da","ta","sa","wa","la",
        "pa","dha","ja","ya","nya","ma","ga","ba","tha","nga",
        "taling","tarung","wignyan",
        "_ha","_sa","_pa",                     # pasangan (beside-type)
        "pada_lingsa","pada_lungsi",           # punctuation — appear at base density line
        "unlabeled",                           # unknown / for later review
    ],
    "sandhangan": [
        "wulu","suku","pepet",
        "layar","cecak","cakra","cakra_keret",
        "cakra_suku","pangkon","pa_cerek","nga_lelet",
        "adeg_adeg",
        "_nya","_na","_wa",                     # pasangan (other positions)
        "unlabeled",                           # unknown / for later review
    ],
}

# Internal positions used to read crops from density_processing output.
# Keep all 5 so crop keys from density_processing still resolve correctly.
POSITIONS = ["base", "above", "below", "beside", "wrapped"]

# Maps each density_processing crop position → training bucket.
POSITION_GROUP: dict[str, str] = {
    "base"   : "base",
    "above"  : "sandhangan",
    "below"  : "sandhangan",
    "beside" : "sandhangan",
    "wrapped": "sandhangan",
}

# Directory for optional custom glyph images (fallback for buttons).
GLYPH_IMAGES_DIR = os.path.join(script_dir, "glyph_images")

def scan_glyph_images() -> dict[str, dict[str, str]]:
    """Scan glyph_images/{base,sandhangan}/ for .png files matching class names."""
    result: dict[str, dict[str, str]] = {}
    for bucket in POSITION_CLASSES:
        bucket_dir = os.path.join(GLYPH_IMAGES_DIR, bucket)
        if not os.path.isdir(bucket_dir):
            continue
        for fname in os.listdir(bucket_dir):
            name, ext = os.path.splitext(fname)
            if ext.lower() == ".png" and name in POSITION_CLASSES[bucket]:
                result[name] = f"/glyph_images/{bucket}/{fname}"
    return result


def maybe_save_glyph(bucket: str, cls: str, img_path: str) -> str | None:
    """Save img_path as glyph_images/{bucket}/{cls}.png if no glyph exists yet.
    Returns the web-relative path if a new glyph was written, else None."""
    glyph_dir  = os.path.join(GLYPH_IMAGES_DIR, bucket)
    glyph_path = os.path.join(glyph_dir, f"{cls}.png")
    if os.path.exists(glyph_path):
        return None
    img = cv2.imread(img_path)
    if img is None:
        return None
    os.makedirs(glyph_dir, exist_ok=True)
    cv2.imwrite(glyph_path, img)
    return f"/glyph_images/{bucket}/{cls}.png"

# ── Helpers ───────────────────────────────────────────────────────────────────

def web_to_disk(web_path: str) -> str:
    """Convert density_processing web-relative path to actual disk path."""
    return os.path.join(CROPS_TMP, os.path.basename(web_path))


def read_b64(disk_path: str) -> str | None:
    """Read an image from disk and return as base64 JPEG string, or None."""
    if not disk_path or not os.path.exists(disk_path):
        if disk_path:
            print(f"  [read_b64] FILE NOT FOUND: {disk_path}")
        return None
    img = cv2.imread(disk_path)
    if img is None:
        print(f"  [read_b64] cv2.imread FAILED for existing file: {disk_path}")
        return None
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def segment_file(fname: str) -> dict:
    """Run density segmentation on fname (cached per session)."""
    if fname not in _seg_cache:
        img_path = os.path.join(INPUT_DIR, fname)
        result = process_image_density_guided(img_path, CROPS_TMP)
        _seg_cache[fname] = result or {}
    return _seg_cache[fname]


def count_labeled() -> dict[str, int]:
    """Count saved crops per label/class in the dataset directory."""
    counts: dict[str, int] = {}
    if MODE == "stage1":
        for label in STAGE1_LABELS:
            d = os.path.join(DATASET_DIR, "dataset_binary", label)
            counts[label] = len([f for f in os.listdir(d) if f.endswith((".jpg", ".png"))]) if os.path.exists(d) else 0
    elif MODE == "stage2":
        for bucket, classes in POSITION_CLASSES.items():
            for cls in classes:
                key = f"{bucket}/{cls}"
                d = os.path.join(DATASET_DIR, "dataset_stage2", bucket, cls)
                counts[key] = len([f for f in os.listdir(d) if f.endswith((".jpg", ".png"))]) if os.path.exists(d) else 0
    else:  # review
        if REVIEW_FOLDER:
            parts = REVIEW_FOLDER.split("/", 1)
            if len(parts) == 2:
                bucket, cls = parts
                d = os.path.join(DATASET_DIR, "dataset_stage2", bucket, cls)
                counts[f"{bucket}/{cls}"] = len([f for f in os.listdir(d) if f.endswith((".jpg", ".png"))]) if os.path.exists(d) else 0
        else:
            for bucket in ["base", "sandhangan"]:
                d = os.path.join(DATASET_DIR, "dataset_stage2", bucket, "unlabeled")
                counts[f"{bucket}/unlabeled"] = len([f for f in os.listdir(d) if f.endswith((".jpg", ".png"))]) if os.path.exists(d) else 0
    return counts


import json

def get_progress_file():
    if MODE == "review" and REVIEW_FOLDER:
        safe = REVIEW_FOLDER.replace("/", "_")
        return os.path.join(DATASET_DIR, f"progress_review_{safe}.json")
    return os.path.join(DATASET_DIR, f"progress_{MODE}.json")

def read_progress():
    pfile = get_progress_file()
    if os.path.exists(pfile):
        try:
            with open(pfile, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"file_idx": 0, "char_idx": 0, "pos_idx": 0, "label_pass": 0}

def save_progress(file_idx, char_idx=0, pos_idx=0, label_pass=0):
    pfile = get_progress_file()
    try:
        with open(pfile, "w") as f:
            json.dump({"file_idx": file_idx, "char_idx": char_idx, "pos_idx": pos_idx, "label_pass": label_pass}, f)
    except Exception:
        pass

def ensure_dirs():
    os.makedirs(CROPS_TMP, exist_ok=True)
    if MODE == "stage1":
        for label in STAGE1_LABELS:
            os.makedirs(os.path.join(DATASET_DIR, "dataset_binary", label), exist_ok=True)
    elif MODE == "stage2":
        for bucket, classes in POSITION_CLASSES.items():
            for cls in classes:
                os.makedirs(os.path.join(DATASET_DIR, "dataset_stage2", bucket, cls), exist_ok=True)
    # review: dataset_stage2 dirs are expected to exist from stage2


def scan_review_items() -> list:
    """Collect crops for review mode.
    If REVIEW_FOLDER is set (e.g. 'sandhangan/wulu'), scan that specific folder.
    Otherwise scan all unlabeled/ folders."""
    items = []
    exts  = (".png", ".jpg", ".jpeg")

    if REVIEW_FOLDER:
        parts = REVIEW_FOLDER.split("/", 1)
        if len(parts) == 2:
            bucket, cls = parts
            d = os.path.join(DATASET_DIR, "dataset_stage2", bucket, cls)
            if os.path.isdir(d):
                for fname in sorted(os.listdir(d)):
                    if fname.lower().endswith(exts):
                        items.append({"bucket": bucket, "disk_path": os.path.join(d, fname), "fname": fname})
    else:
        for bucket in ["base", "sandhangan"]:
            d = os.path.join(DATASET_DIR, "dataset_stage2", bucket, "unlabeled")
            if not os.path.isdir(d):
                continue
            for fname in sorted(os.listdir(d)):
                if fname.lower().endswith(exts):
                    items.append({"bucket": bucket, "disk_path": os.path.join(d, fname), "fname": fname})
    return items


def load_custom_classes():
    """Merge custom_classes.json into POSITION_CLASSES (called at startup)."""
    if not CUSTOM_CLASSES_FILE or not os.path.exists(CUSTOM_CLASSES_FILE):
        return
    try:
        with open(CUSTOM_CLASSES_FILE, "r") as f:
            custom = json.load(f)
        for bucket, classes in custom.items():
            if bucket not in POSITION_CLASSES:
                continue
            for cls in classes:
                if cls not in POSITION_CLASSES[bucket]:
                    if "unlabeled" in POSITION_CLASSES[bucket]:
                        idx = POSITION_CLASSES[bucket].index("unlabeled")
                        POSITION_CLASSES[bucket].insert(idx, cls)
                    else:
                        POSITION_CLASSES[bucket].append(cls)
        print(f"  Custom  : loaded {sum(len(v) for v in custom.values())} extra class(es) from custom_classes.json")
    except Exception as e:
        print(f"  [warn] Could not load custom_classes.json: {e}")


def persist_custom_class(bucket: str, cls: str):
    """Append a new class to custom_classes.json so it survives restarts."""
    if not CUSTOM_CLASSES_FILE:
        return
    custom: dict = {}
    if os.path.exists(CUSTOM_CLASSES_FILE):
        try:
            with open(CUSTOM_CLASSES_FILE, "r") as f:
                custom = json.load(f)
        except Exception:
            pass
    custom.setdefault(bucket, [])
    if cls not in custom[bucket]:
        custom[bucket].append(cls)
    try:
        with open(CUSTOM_CLASSES_FILE, "w") as f:
            json.dump(custom, f, indent=2)
    except Exception as e:
        print(f"  [warn] Could not save custom_classes.json: {e}")


# ── Labeling goals ───────────────────────────────────────────────────────────

LABELING_GOALS: dict[str, int] = {
    # Priority 1 — fix failing sandhangan (→ 60 each)
    "sandhangan/pepet":      60,
    "sandhangan/layar":      60,
    "sandhangan/pangkon":    60,
    "sandhangan/_ta":        60,
    "sandhangan/_dha":       60,
    "sandhangan/_ka":        60,
    "sandhangan/_na":        60,
    "sandhangan/cecak":      60,
    "sandhangan/wulu_cecak": 60,
    # Priority 2 — bring dropped sandhangan above threshold (→ 20 each)
    "sandhangan/_ja":        20,
    "sandhangan/_da":        20,
    "sandhangan/pengkal":    20,
    "sandhangan/_ba":        20,
    "sandhangan/_wa":        20,
    "sandhangan/_ca":        20,
    "sandhangan/_ya":        20,
    "sandhangan/_la":        20,
    "sandhangan/_nga":       20,
    # Priority 3 — rescue dropped base classes (→ 20) and boost small ones (→ 30)
    "base/ca":          20,
    "base/dha":         20,
    "base/ga":          20,
    "base/pada_lungsi": 20,
    "base/ba":          30,
    "base/ja":          30,
    "base/nya":         30,
    "base/_ha":         30,
    "base/_pa":         30,
    "base/tarung":      30,
    "base/wignyan":     30,
}

GOAL_PRIORITIES = [
    {
        "label": "P1 — Fix failing sandhangan",
        "keys": [
            "sandhangan/pepet", "sandhangan/layar", "sandhangan/pangkon",
            "sandhangan/_ta", "sandhangan/_dha", "sandhangan/_ka",
            "sandhangan/_na", "sandhangan/cecak", "sandhangan/wulu_cecak",
        ],
    },
    {
        "label": "P2 — Bring dropped sandhangan in",
        "keys": [
            "sandhangan/_ja", "sandhangan/_da", "sandhangan/pengkal",
            "sandhangan/_ba", "sandhangan/_wa", "sandhangan/_ca",
            "sandhangan/_ya", "sandhangan/_la", "sandhangan/_nga",
        ],
    },
    {
        "label": "P3 — Rescue / boost base classes",
        "keys": [
            "base/ca", "base/dha", "base/ga", "base/pada_lungsi",
            "base/ba", "base/ja", "base/nya",
            "base/_ha", "base/_pa", "base/tarung", "base/wignyan",
        ],
    },
]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("density_labeler.html")


@app.route("/glyph_images/<path:filename>")
def serve_glyph_image(filename):
    return send_from_directory(GLYPH_IMAGES_DIR, filename)


@app.route("/api/config")
def api_config():
    return jsonify({
        "mode"            : MODE,
        "review_folder"   : REVIEW_FOLDER,
        "positions"       : POSITIONS,
        "position_classes": POSITION_CLASSES,
        "position_group"  : POSITION_GROUP,
        "stage1_labels"   : STAGE1_LABELS,
        "glyph_images"    : scan_glyph_images(),
    })


@app.route("/api/init")
def api_init():
    return jsonify({
        "total_files"       : len(ALL_FILES),
        "total_review_items": len(REVIEW_ITEMS),
        "counts"            : count_labeled(),
        "mode"              : MODE,
        "progress"          : read_progress(),
    })

@app.route("/api/goal_progress")
def api_goal_progress():
    priorities = []
    for p in GOAL_PRIORITIES:
        classes = []
        for key in p["keys"]:
            bucket, cls = key.split("/", 1)
            d = os.path.join(DATASET_DIR, "dataset_stage2", bucket, cls)
            current = len([f for f in os.listdir(d) if f.endswith((".jpg", ".png"))]) if os.path.isdir(d) else 0
            goal = LABELING_GOALS.get(key, 20)
            classes.append({"key": key, "name": cls, "current": current, "goal": goal, "done": current >= goal})
        priorities.append({
            "label": p["label"],
            "classes": classes,
            "done": sum(1 for c in classes if c["done"]),
            "total": len(classes),
        })
    return jsonify({"priorities": priorities})


@app.route("/api/progress", methods=["POST"])
def api_progress():
    data = request.json
    save_progress(data.get("file_idx", 0), data.get("char_idx", 0), data.get("pos_idx", 0), data.get("label_pass", 0))
    return jsonify({"status": "ok"})


@app.route("/api/load_review_item")
def api_load_review_item():
    """Return the next available unlabeled crop starting from item_idx."""
    if not REVIEW_ITEMS:
        return jsonify({"error": "no_items"})

    item_idx = int(request.args.get("item_idx", 0))
    if item_idx < 0:
        item_idx = 0

    # Scan forward, skipping items already moved or unreadable
    while item_idx < len(REVIEW_ITEMS):
        item = REVIEW_ITEMS[item_idx]
        if not os.path.exists(item["disk_path"]):
            item_idx += 1
            continue
        b64 = read_b64(item["disk_path"])
        if b64 is None:
            item_idx += 1
            continue
        remaining = sum(1 for i in REVIEW_ITEMS if os.path.exists(i["disk_path"]))
        return jsonify({
            "item_idx" : item_idx,
            "total"    : len(REVIEW_ITEMS),
            "remaining": remaining,
            "bucket"   : item["bucket"],
            "disk_path": item["disk_path"],
            "fname"    : item["fname"],
            "b64"      : b64,
        })

    return jsonify({"error": "eof"})


@app.route("/api/add_class", methods=["POST"])
def api_add_class():
    """Register a new class for a bucket at runtime and persist it to custom_classes.json."""
    data   = request.json
    bucket = data.get("bucket", "").strip()
    cls    = data.get("class_name", "").strip()

    if not bucket or not cls:
        return jsonify({"error": "missing bucket or class_name"})
    if bucket not in POSITION_CLASSES:
        return jsonify({"error": f"unknown bucket '{bucket}'"})

    added = False
    if cls not in POSITION_CLASSES[bucket]:
        if "unlabeled" in POSITION_CLASSES[bucket]:
            idx = POSITION_CLASSES[bucket].index("unlabeled")
            POSITION_CLASSES[bucket].insert(idx, cls)
        else:
            POSITION_CLASSES[bucket].append(cls)
        os.makedirs(os.path.join(DATASET_DIR, "dataset_stage2", bucket, cls), exist_ok=True)
        persist_custom_class(bucket, cls)
        added = True
        print(f"  [add_class] '{cls}' → {bucket} (new)")

    return jsonify({"status": "ok", "class_name": cls, "bucket": bucket, "added": added})


@app.route("/api/load_file")
def api_load_file():
    file_idx = int(request.args.get("file_idx", 0))
    if file_idx < 0 or file_idx >= len(ALL_FILES):
        return jsonify({"error": "eof"})

    fname  = ALL_FILES[file_idx]
    result = segment_file(fname)

    if not result or not result.get("cropped_results"):
        return jsonify({"error": "segment_failed", "filename": fname})

    # Read original line image for display
    img = cv2.imread(os.path.join(INPUT_DIR, fname))
    if img is None:
        return jsonify({"error": "missing"})
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    line_b64 = base64.b64encode(buf).decode("utf-8")

    # Build per-character data
    characters = []
    boxes = []
    for ci, cr in enumerate(result["cropped_results"]):
        if "box" in cr:
            boxes.append(cr["box"])
        char_data: dict[str, dict | None] = {}
        avail_positions = []
        for pos in POSITIONS:
            web_path = cr.get(pos)
            if web_path:
                disk = web_to_disk(web_path)
                b64  = read_b64(disk)
                box  = cr.get("boxes", {}).get(pos)
                char_data[pos] = {"b64": b64, "disk_path": disk, "box": box} if b64 else None
                if b64:
                    avail_positions.append(pos)
            else:
                char_data[pos] = None
        print(f"  char[{ci}] positions: {avail_positions}")
        characters.append(char_data)

    return jsonify({
        "filename"       : fname,
        "image"          : line_b64,
        "original_width" : int(img.shape[1]),
        "original_height": int(img.shape[0]),
        "characters"     : characters,
        "boxes"          : boxes,
        "total_chars"    : len(characters),
    })


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.json

    # Review mode: move crop from unlabeled/ to the chosen class folder
    if MODE == "review":
        bucket      = data["bucket"]
        class_label = data["class_label"]
        src_path    = data["disk_path"]
        fname       = os.path.basename(src_path)
        dest_dir    = os.path.join(DATASET_DIR, "dataset_stage2", bucket, class_label)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path   = os.path.join(dest_dir, fname)
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
        moved = False
        if os.path.exists(src_path) and src_path != dest_path:
            shutil.move(src_path, dest_path)
            moved = True
        glyph_web = maybe_save_glyph(bucket, class_label, dest_path)
        return jsonify({
            "status"       : "ok",
            "saved_path"   : dest_path,
            "label_key"    : f"{bucket}/{class_label}",
            "src_path"     : src_path,
            "moved"        : moved,
            "glyph_created": glyph_web,
        })

    # Stage 1 / Stage 2
    fname     = data["filename"]
    char_idx  = int(data["char_idx"])
    disk_path = data["disk_path"]

    base_name = os.path.splitext(fname)[0]
    crop_name = os.path.basename(disk_path)
    save_name = f"{base_name}_c{char_idx:03d}_{crop_name}"

    if MODE == "stage1":
        label    = data["label"]
        dest_dir = os.path.join(DATASET_DIR, "dataset_binary", label)
        label_key = label
    else:
        position    = data["position"]
        class_label = data["class_label"]
        bucket      = POSITION_GROUP[position]  # "base" or "sandhangan"
        dest_dir    = os.path.join(DATASET_DIR, "dataset_stage2", bucket, class_label)
        label_key   = f"{bucket}/{class_label}"

    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, save_name)

    if os.path.exists(disk_path):
        shutil.copy2(disk_path, dest_path)

    glyph_web = None
    if MODE == "stage2":
        glyph_web = maybe_save_glyph(bucket, class_label, dest_path)

    return jsonify({"status": "ok", "saved_path": dest_path, "label_key": label_key, "glyph_created": glyph_web})


@app.route("/api/debug_char")
def api_debug_char():
    """Debug endpoint: show raw segmentation data for a specific character."""
    file_idx = int(request.args.get("file_idx", 0))
    char_idx = int(request.args.get("char_idx", 0))
    if file_idx < 0 or file_idx >= len(ALL_FILES):
        return jsonify({"error": "invalid file_idx"})
    fname = ALL_FILES[file_idx]
    result = segment_file(fname)
    if not result or not result.get("cropped_results"):
        return jsonify({"error": "no segmentation"})
    crs = result["cropped_results"]
    if char_idx < 0 or char_idx >= len(crs):
        return jsonify({"error": "invalid char_idx", "total_chars": len(crs)})
    cr = crs[char_idx]
    debug_info = {"filename": fname, "char_idx": char_idx}
    for pos in POSITIONS:
        web_path = cr.get(pos)
        if web_path:
            disk = web_to_disk(web_path)
            debug_info[pos] = {
                "web_path": web_path,
                "disk_path": disk,
                "disk_exists": os.path.exists(disk),
                "readable": cv2.imread(disk) is not None if os.path.exists(disk) else False,
            }
        else:
            debug_info[pos] = None
    return jsonify(debug_info)


@app.route("/api/undo", methods=["POST"])
def api_undo():
    data         = request.json
    path         = data.get("path", "")
    restore_path = data.get("restore_path", "")  # review mode: move back to unlabeled
    if restore_path and path and path != restore_path and os.path.exists(path):
        os.makedirs(os.path.dirname(restore_path), exist_ok=True)
        shutil.move(path, restore_path)
    elif path and os.path.exists(path) and not restore_path:
        os.remove(path)
    return jsonify({"status": "ok"})


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global INPUT_DIR, DATASET_DIR, CROPS_TMP, MODE, ALL_FILES, CUSTOM_CLASSES_FILE, REVIEW_FOLDER

    parser = argparse.ArgumentParser(description="Density-guided Javanese OCR Labeler")
    parser.add_argument("--input",  "-i", default=".",
                        help="Directory of line images to segment and label (not used in review mode)")
    parser.add_argument("--output", "-o", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Root directory for dataset output (default: script dir)")
    parser.add_argument("--crops",  "-c", default="labeler_crops",
                        help="Temp dir where density_processing saves crop files (default: labeler_crops/)")
    parser.add_argument("--mode",   "-m", choices=["stage1", "stage2", "review"], default="stage1",
                        help="stage1 = valid/noise for base crops | stage2 = class per position | review = relabel unlabeled crops")
    parser.add_argument("--folder", "-f", default="",
                        help="Review a specific labeled folder instead of unlabeled/ (e.g. sandhangan/wulu). Only used with --mode review.")
    parser.add_argument("--port",   "-p", type=int, default=5005)
    args = parser.parse_args()

    INPUT_DIR           = os.path.abspath(args.input)
    DATASET_DIR         = os.path.abspath(args.output)
    CROPS_TMP           = os.path.abspath(args.crops)
    MODE                = args.mode
    REVIEW_FOLDER       = args.folder.strip("/") if args.folder else ""
    CUSTOM_CLASSES_FILE = os.path.join(DATASET_DIR, "custom_classes.json")

    load_custom_classes()
    ensure_dirs()

    print(f"\n{'─'*50}")
    print(f"  Density Labeler  |  Mode: {MODE.upper()}")
    print(f"{'─'*50}")

    if MODE == "review":
        REVIEW_ITEMS.clear()
        REVIEW_ITEMS.extend(scan_review_items())
        print(f"  Dataset : {DATASET_DIR}")
        if REVIEW_FOLDER:
            print(f"  Review  : {len(REVIEW_ITEMS)} items in dataset_stage2/{REVIEW_FOLDER}/")
        else:
            print(f"  Review  : {len(REVIEW_ITEMS)} unlabeled items")
            base_cnt  = sum(1 for i in REVIEW_ITEMS if i["bucket"] == "base")
            sandh_cnt = sum(1 for i in REVIEW_ITEMS if i["bucket"] == "sandhangan")
            print(f"            base={base_cnt}  sandhangan={sandh_cnt}")
    else:
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        ALL_FILES = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(exts)])
        print(f"  Input   : {INPUT_DIR}  ({len(ALL_FILES)} images)")
        print(f"  Dataset : {DATASET_DIR}")
        print(f"  Crops   : {CROPS_TMP}")

    print(f"  → Open  : http://127.0.0.1:{args.port}")
    print(f"{'─'*50}\n")

    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
