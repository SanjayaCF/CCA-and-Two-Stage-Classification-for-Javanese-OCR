# SegmentationTuner

A visual testing tool for adjusting the density-guided CCA segmentation parameters used in AksaraOCR. It was used during development to find the right threshold and connectivity values before they were fixed in the main pipeline.

Part of the thesis "Implementasi Connected Component Analysis dan Klasifikasi Dua Tahap untuk OCR Aksara Jawa" at Universitas Kristen Duta Wacana (UKDW), 2026.

## What you can tune

- Density threshold for the horizontal and vertical profile analysis
- CCA connectivity (4-connected vs. 8-connected)
- Minimum and maximum component size filters
- Character bounding box expansion margins

The browser interface shows the segmentation result on each test image in real time as you adjust the values. The current parameter state is saved to `test_state.json` between sessions.

## Requirements

```
pip install flask pillow numpy
```

## Running

```
python app.py
```

Open `http://localhost:5002` in your browser, or whichever port is set in `app.py`.

## Files

| File | Description |
|---|---|
| `app.py` | main application |
| `test_state.json` | last saved parameter configuration |
| `templates/` | HTML templates |

## Related repositories

| Repository | Description |
|---|---|
| [AksaraOCR](../AksaraOCR) | the pipeline that uses the finalized parameters |
| [DensityLabeler](../DensityLabeler) | labeling tool |
