import os
import json
import random
import sys
import glob
import csv
from io import StringIO
from flask import Flask, render_template, request, jsonify, send_from_directory, Response

# Extend sys.path to find processing scripts in Kerja Praktik
KP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Kerja Praktik'))
sys.path.append(KP_DIR)

try:
    from processing import process_image
    from new_processing import process_image_v2
except ImportError as e:
    print(f"Error importing from Kerja Praktik: {e}")
    sys.exit(1)

app = Flask(__name__)

# Config
DATA_DIR = os.path.join(KP_DIR, 'Y2outputs', 'luaran')
STATE_FILE = os.path.join(os.path.dirname(__file__), 'test_state.json')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'static', 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

# Application state
state = {
    'images': [],
    'accepted_old': {},
    'accepted_new': {},
    'ignored': []
}

def load_state():
    global state
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            loaded = json.load(f)
            state['images'] = loaded.get('images', [])
            state['accepted_old'] = loaded.get('accepted_old', {})
            state['accepted_new'] = loaded.get('accepted_new', {})
            state['ignored'] = loaded.get('ignored', [])
    else:
        all_images = glob.glob(os.path.join(DATA_DIR, '*.jpg'))
        if not all_images:
            all_images = glob.glob(os.path.join(DATA_DIR, '*.png'))
        
        all_images = sorted([os.path.basename(p) for p in all_images])
        state['images'] = all_images[:100]
        state['accepted_old'] = {}
        state['accepted_new'] = {}
        state['ignored'] = []
        save_state()

def save_state():
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

load_state()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/state', methods=['GET'])
def get_state():
    return jsonify(state)

@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(DATA_DIR, filename)

@app.route('/process', methods=['POST'])
def process_single():
    data = request.json
    filename = data.get('filename')
    method = data.get('method')
    
    input_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(input_path):
        return jsonify({'error': 'Image not found'}), 404

    try:
        if method == 'old':
            result = process_image(
                input_path, RESULT_DIR,
                threshold=int(data.get('threshold', 160)),
                kernel_size=int(data.get('kernel_size', 5)),
                min_area=int(data.get('min_area', 15)),
                grouping_distance=int(data.get('grouping_distance', 8))
            )
        else:
            result = process_image_v2(
                input_path, RESULT_DIR,
                threshold=int(data.get('threshold', 160)),
                kernel_size=int(data.get('kernel_size', 5)),
                min_area=int(data.get('min_area', 15)),
                main_height_ratio=float(data.get('main_height_ratio', 0.5)),
                main_merge_gap=int(data.get('main_merge_gap', 5))
            )
        
        char_count = len(result.get('cropped_results', []))
        return jsonify({'result': result, 'char_count': char_count})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/accept', methods=['POST'])
def accept_image():
    data = request.json
    filename = data.get('filename')
    method = data.get('method')
    expected = data.get('expected')
    params = data.get('params')

    if filename in state['ignored']:
        state['ignored'].remove(filename)

    if method == 'old':
        state['accepted_old'][filename] = {'expected': expected, 'params': params}
    else:
        state['accepted_new'][filename] = {'expected': expected, 'params': params}
    
    save_state()
    return jsonify({'status': 'success'})

@app.route('/ignore', methods=['POST'])
def ignore_image():
    data = request.json
    filename = data.get('filename')
    if filename not in state['ignored']:
        state['ignored'].append(filename)
    
    if filename in state['accepted_old']:
        del state['accepted_old'][filename]
    if filename in state['accepted_new']:
        del state['accepted_new'][filename]

    save_state()
    return jsonify({'status': 'success'})

@app.route('/revalidate', methods=['POST'])
def revalidate():
    data = request.json
    method = data.get('method')
    current_params = data.get('params')
    
    accepted_dict = state['accepted_old'] if method == 'old' else state['accepted_new']
    failures = []

    for filename, info in accepted_dict.items():
        if filename in state['ignored']:
            continue

        expected = info['expected']
        input_path = os.path.join(DATA_DIR, filename)
        
        def test_with_thresh(t):
            try:
                p = dict(current_params)
                p['threshold'] = t
                if method == 'old':
                    res = process_image(
                        input_path, RESULT_DIR,
                        threshold=t,
                        kernel_size=int(p.get('kernel_size', 5)),
                        min_area=int(p.get('min_area', 15)),
                        grouping_distance=int(p.get('grouping_distance', 8))
                    )
                else:
                    res = process_image_v2(
                        input_path, RESULT_DIR,
                        threshold=t,
                        kernel_size=int(p.get('kernel_size', 5)),
                        min_area=int(p.get('min_area', 15)),
                        main_height_ratio=float(p.get('main_height_ratio', 0.5)),
                        main_merge_gap=int(p.get('main_merge_gap', 5))
                    )
                return len(res.get('cropped_results', [])) == expected
            except:
                return False

        initial_test = test_with_thresh(int(current_params['threshold']))
        
        if not initial_test:
            success_thresholds = []
            for t in range(50, 255, 10):
                if test_with_thresh(t):
                    success_thresholds.append(t)
            
            if len(success_thresholds) > 0:
                local_min = min(success_thresholds)
                local_max = max(success_thresholds)
                failures.append({
                    'filename': filename,
                    'msg': f"Failed with Threshold {current_params['threshold']}. Safe range for this image: {local_min} - {local_max}"
                })
            else:
                failures.append({
                    'filename': filename,
                    'msg': f"Failed with Threshold {current_params['threshold']}. Could not find ANY safe threshold with these other parameters."
                })

    return jsonify({
        'failures': failures
    })

@app.route('/export_csv', methods=['GET'])
def export_csv():
    # Merge old and new states into one CSV representation
    si = StringIO()
    writer = csv.writer(si)
    
    headers = [
        'Filename', 'Ignored', 
        'Old_Expected', 'Old_Threshold', 'Old_Kernel', 'Old_MinArea', 'Old_GDist',
        'New_Expected', 'New_Threshold', 'New_Kernel', 'New_MinArea', 'New_MHR', 'New_MMG'
    ]
    writer.writerow(headers)
    
    for filename in state['images']:
        ignored = 'Yes' if filename in state['ignored'] else 'No'
        row = [filename, ignored]
        
        if filename in state['accepted_old']:
            o = state['accepted_old'][filename]
            p = o['params']
            row.extend([o['expected'], p.get('threshold'), p.get('kernel_size'), p.get('min_area'), p.get('grouping_distance')])
        else:
            row.extend(['', '', '', '', ''])

        if filename in state['accepted_new']:
            n = state['accepted_new'][filename]
            p = n['params']
            row.extend([n['expected'], p.get('threshold'), p.get('kernel_size'), p.get('min_area'), p.get('main_height_ratio'), p.get('main_merge_gap')])
        else:
            row.extend(['', '', '', '', '', ''])
            
        writer.writerow(row)
    
    output = si.getvalue()
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=regression_test_export.csv"}
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001)
