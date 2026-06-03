# Greenscale

Greenscale is an image-analysis and machine-learning pipeline for estimating root biomass from plant images.

The project combines classical computer vision techniques (adaptive thresholding, Frangi filtering, and morphological operations) with deep learning to generate root masks and predict root weight.

This public repository is a redacted demonstration of techniques used in agricultural image analysis workflows.

## Workflow

Raw Images
    ↓
Image Binarization
    ↓
Mask Evaluation
    ↓
Best Method Selection
    ↓
Target Map Generation
    ↓
Neural Network Training
    ↓
Root Weight Prediction

This repository is a public, redaction‑safe sample of a company program that employs a image recognition model to analyze plant root images. It demonstrates sample code without exposing any proprietary logic.

✅ You can share this repo publicly. Proprietary parsing rules, vendor/OCR config, and corp data are not included.

## Scripts

```bash

python scripts/download_data.py
python scripts/create_masks.py
python scripts/evaluate_masks.py
python scripts/pipeline.py
python scripts/train.py
python scripts/predict.py
python scripts/run_workflow.py

```


### File Structure

<pre>
.
├── src/
├── tests/
├── docs/
├── LICENSE
├── README.md
└── CONTRIBUTING.md
