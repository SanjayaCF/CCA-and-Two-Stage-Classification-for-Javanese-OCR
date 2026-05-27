# Javanese Script OCR

Source code repositories for the thesis "Implementasi Connected Component Analysis dan Klasifikasi Dua Tahap untuk OCR Aksara Jawa" (Implementation of Connected Component Analysis and Two-Stage Classification for Javanese Script OCR) at Universitas Kristen Duta Wacana (UKDW), 2026.

Author: Sanjaya Cahyadi Fuad (71220965)

---

## AksaraOCR

[AksaraOCR](./AksaraOCR) is the main project. It is a Flask web application that takes a scanned Javanese document image and returns the recognized text in Unicode Aksara Jawa.

The pipeline works in four stages:

**Segmentation.** Rather than using standard connected component analysis directly on the binarized image, the system first computes horizontal and vertical density profiles for each candidate region. This makes it possible to separate characters that are physically connected in print, such as base characters with their pasangan or stacked sandhangan. The result is a set of individually bounded character components.

**Stage 1 filtering.** Each component is classified by a binary EfficientNetB0 model that was trained to distinguish valid Aksara Jawa characters from segmentation noise. Components that do not pass this filter are discarded before recognition.

**Stage 2 recognition.** Valid components are sent to one of two separate EfficientNetB0 classifiers depending on their estimated position in the character cell. The base classifier covers 27 classes (the hanacaraka consonants, common pasangan forms, and punctuation marks). The sandhangan classifier covers 23 classes (vowel modifiers, cecak, layar, and compound diacritics).

**Composition.** The recognized labels are assembled into Unicode text using a rule-based composer that handles pasangan attachment, sandhangan ordering, and pada punctuation following the conventions of printed Aksara Jawa.

To run the system, install the requirements, place the trained models in `tahap1/` and `tahap2/`, then run `python app.py`. The models can be trained from scratch using [TwoPhaseClassifier](./TwoPhaseClassifier), or requested from the author.

---

## Supporting repositories

### TwoPhaseClassifier

[TwoPhaseClassifier](./TwoPhaseClassifier) contains the Jupyter notebooks used to train the Stage 1 and Stage 2 models. If you want to retrain from scratch or review the training results, start here.

The Stage 1 notebook trains the binary classifier using two-phase fine-tuning and records the threshold selection process. The Stage 2 notebook trains the base and sandhangan classifiers independently, since their character sets have different visual properties that respond better to separate training configurations.

### DensityLabeler

[DensityLabeler](./DensityLabeler) is the browser-based tool used to build the training dataset. It displays each character crop next to its density profile, which helps distinguish visually similar characters during labeling. The tool supports both Stage 1 labeling and per-class Stage 2 labeling with progress tracking.

The full dataset (around 25,000 images) is not included in the repository due to size. Contact the author if you need access to it.

### PerformanceTesting

[PerformanceTesting](./PerformanceTesting) is the review interface used to measure the pipeline on 50 sampled test images. A reviewer goes through each image, marks the OCR output as correct, wrong, or a segmentation error, and the app computes segmentation accuracy, over-segmentation rate, character recognition accuracy, and end-to-end accuracy from those annotations.

The saved results from the evaluation are included in `eval_results.json`.

### SegmentationTuner

[SegmentationTuner](./SegmentationTuner) is a development tool that was used to tune the CCA segmentation parameters before they were finalized. It lets you adjust density thresholds, connectivity settings, and component size filters while seeing the segmentation output update in real time on test images.

---

## Repositories

| Repository | Description |
|---|---|
| [AksaraOCR](./AksaraOCR) | main OCR web application |
| [TwoPhaseClassifier](./TwoPhaseClassifier) | CNN training notebooks (Stage 1 and Stage 2) |
| [DensityLabeler](./DensityLabeler) | dataset labeling tool |
| [PerformanceTesting](./PerformanceTesting) | end-to-end evaluation interface |
| [SegmentationTuner](./SegmentationTuner) | segmentation parameter tuning tool |
