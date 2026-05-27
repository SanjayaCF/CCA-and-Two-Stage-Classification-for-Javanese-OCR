# AksaraOCR

AksaraOCR is a web-based OCR system for printed Javanese (Aksara Jawa) script. It segments characters from document images using a density-guided Connected Component Analysis approach, then classifies them through a two-stage CNN pipeline to produce Unicode output.

Built as part of the thesis "Implementasi Connected Component Analysis dan Klasifikasi Dua Tahap untuk OCR Aksara Jawa" at Universitas Kristen Duta Wacana (UKDW), 2026.

## How it works

The system processes a scanned document image through four stages. First, density-guided CCA extracts individual character components from the image. Each component is then passed to a binary EfficientNetB0 classifier that filters out noise and non-character regions. The remaining valid components go through a second stage with two separate models: one for base characters (27 classes) and one for diacritics and sandhangan (23 classes). Finally, the recognized components are assembled into Unicode text following Javanese script composition rules.

## Evaluation

Tested on 50 document images containing 441 ground-truth characters:

| Metric | Result |
|---|---|
| Segmentation Accuracy | 95.46% |
| Over-segmentation Rate | 3.88% |
| Character Recognition Accuracy | 92.87% |
| End-to-End Accuracy | 88.66% |

## Setup

```
pip install -r requirements.txt
```

The trained model files (.keras) belong in `tahap1/` and `tahap2/`. You can train them from scratch using the [TwoPhaseClassifier](../TwoPhaseClassifier) notebooks, or contact the author for the weights directly.

## Running

```
python app.py
```

Then open `http://localhost:5000` in your browser and upload a document image to start.

## Project structure

```
AksaraOCR/
├── app.py
├── requirements.txt
├── processing/
│   ├── stage1.py       # binary classifier (valid character vs. noise)
│   ├── stage2.py       # character recognition (base + sandhangan)
│   └── composition.py  # Unicode assembly with Javanese script rules
├── templates/
└── static/             # stylesheet and the CARAKAN JAWA font
```

## Related repositories

| Repository | Description |
|---|---|
| [TwoPhaseClassifier](../TwoPhaseClassifier) | training notebooks for the Stage 1 and Stage 2 models |
| [DensityLabeler](../DensityLabeler) | web tool for building and labeling the training dataset |
| [PerformanceTesting](../PerformanceTesting) | end-to-end evaluation interface |
| [SegmentationTuner](../SegmentationTuner) | visual tool for tuning segmentation parameters |
