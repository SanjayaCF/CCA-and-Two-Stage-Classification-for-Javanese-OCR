# TwoPhaseClassifier

Training notebooks for the two-stage CNN classification models used in AksaraOCR.

Part of the thesis "Implementasi Connected Component Analysis dan Klasifikasi Dua Tahap untuk OCR Aksara Jawa" at Universitas Kristen Duta Wacana (UKDW), 2026.

## Models

### Stage 1 - Binary Classifier

Notebook: `notebook1_stage1_binary_classifier.ipynb`

An EfficientNetB0 model with a Dense(128) head, trained in two phases with class weights to handle the imbalance between valid characters and noise. The decision threshold was tuned to 0.38 based on the test set ROC curve.

Test set results: Accuracy 98.67%, Macro F1 98.50%, AUC-ROC 99.77%

### Stage 2 - Character Recognizer

Notebook: `notebook2_stage2_classifier.ipynb`

Two separate EfficientNetB0 models with Dense(256) heads, one for base characters and one for sandhangan. They are trained independently because the two groups have different visual characteristics that benefit from separate fine-tuning strategies.

- Base classifier (27 classes): val accuracy 98.75%, macro F1 98.61%
- Sandhangan classifier (23 classes): val accuracy 95.53%, macro F1 95.11%

### Inference helper

`inference_stage1.py` is a standalone script for running Stage 1 inference outside the full pipeline.

## Dataset

The training data is built with [DensityLabeler](../DensityLabeler). Before running the notebooks, place the dataset folders alongside them:

```
dataset_binary/
├── valid/       # around 717 valid character crops
└── noise/       # around 70 noise crops

dataset_stage2/
├── base/        # base character crops, 27 classes, around 1463 samples total
└── sandhangan/  # diacritic crops, 23 classes
```

## Requirements

```
pip install tensorflow numpy matplotlib scikit-learn
```

## Files

| File | Description |
|---|---|
| `notebook1_stage1_binary_classifier.ipynb` | Stage 1 training |
| `notebook1_stage1_result.ipynb` | Stage 1 results and evaluation |
| `notebook2_stage2_classifier.ipynb` | Stage 2 training |
| `notebook2_stage2_result.ipynb` | Stage 2 results and evaluation |
| `inference_stage1.py` | standalone Stage 1 inference |

## Related repositories

| Repository | Description |
|---|---|
| [AksaraOCR](../AksaraOCR) | main OCR application that loads these models |
| [DensityLabeler](../DensityLabeler) | tool used to build the training dataset |
