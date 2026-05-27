# DensityLabeler

A browser-based labeling tool for annotating Aksara Jawa character crops. It was used to build the training dataset for AksaraOCR, with a density profile view shown alongside each crop to help identify characters more accurately.

Part of the thesis "Implementasi Connected Component Analysis dan Klasifikasi Dua Tahap untuk OCR Aksara Jawa" at Universitas Kristen Duta Wacana (UKDW), 2026.

## Features

- Side-by-side view of the character crop and its density profile
- Supports both Stage 1 labeling (valid vs. noise) and Stage 2 labeling (27 base classes and 23 sandhangan classes)
- Tracks labeling progress per class
- Can run inference through the OCR pipeline via `inference_compose.py`

## Dataset

The dataset folders are not included here due to size (around 25,000 images, 313 MB total). Contact the author for the labeled dataset.

If you are building the dataset from scratch, the tool expects this structure placed next to the scripts:

```
dataset_binary/
├── valid/
└── noise/

dataset_stage2/
├── base/
└── sandhangan/
```

## Requirements

```
pip install flask pillow numpy
```

## Running

```
python density_labeler.py
```

Open `http://localhost:5001` in your browser.

## Files

| File | Description |
|---|---|
| `density_labeler.py` | main application |
| `inference_compose.py` | runs inference through the full composition pipeline |
| `custom_classes.json` | class definitions for both Stage 1 and Stage 2 |
| `templates/` | HTML templates |
| `glyph_images/` | reference glyph images shown in the labeling interface |

## Related repositories

| Repository | Description |
|---|---|
| [TwoPhaseClassifier](../TwoPhaseClassifier) | training notebooks that use this dataset |
| [AksaraOCR](../AksaraOCR) | the main OCR pipeline |
