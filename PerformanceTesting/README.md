# PerformanceTesting

A review interface for measuring the end-to-end performance of AksaraOCR on a set of sampled test document images.

Part of the thesis "Implementasi Connected Component Analysis dan Klasifikasi Dua Tahap untuk OCR Aksara Jawa" at Universitas Kristen Duta Wacana (UKDW), 2026.

## Results

Reviewed across 50 document images with 441 ground-truth characters:

| Metric | Result |
|---|---|
| Segmentation Accuracy | 95.46% (421/441) |
| Over-segmentation Rate | 3.88% (17/438) |
| Character Recognition Accuracy | 92.87% (391/421) |
| End-to-End Accuracy | 88.66% (391/441) |

The full per-image breakdown is in `eval_results.json`.

## How it works

`select_samples.py` randomly picks 50 test images from the document set. For each image, the app shows the OCR pipeline output alongside the original and lets the reviewer mark each result as correct, wrong, or a segmentation error. Once all images are reviewed, the four metrics above are computed automatically.

## Running

Run the sample selector first if you have not done so:

```
python select_samples.py
```

Then start the app:

```
python app.py
```

Open `http://localhost:5003` in your browser.

## Requirements

```
pip install flask pillow numpy
```

## Files

| File | Description |
|---|---|
| `app.py` | review interface |
| `select_samples.py` | samples 50 test images from the dataset |
| `eval_results.json` | saved results from the 50-image review |
| `templates/` | HTML templates |

## Related repositories

| Repository | Description |
|---|---|
| [AksaraOCR](../AksaraOCR) | the system under evaluation |
