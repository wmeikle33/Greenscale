[![CI](https://github.com/wmeikle33/Greenscale/actions/workflows/ci.yml/badge.svg)](https://github.com/wmeikle33/Greenscale/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

# Greenscale

Greenscale is an image-analysis and machine-learning pipeline for estimating root biomass from plant images.

The project combines classical computer vision techniques (adaptive thresholding, Frangi filtering, and morphological operations) with deep learning to generate root masks and predict root weight.

This public repository is a redacted demonstration of techniques used in agricultural image analysis workflows.

# Repository Scope
This repository is a redacted demonstration of a proprietary computer vision pipeline for root image analysis. It is intended to showcase project structure, engineering practices, testing, CI/CD, and core image-processing workflows while protecting confidential intellectual property.

Included

Public-facing project architecture and package organization
Example image preprocessing and segmentation workflows
Synthetic/demo data generation utilities
Model training and evaluation scaffolding
Unit tests and CI configuration
Documentation and reproducible setup instructions

Excluded

Proprietary datasets and annotations
Production-trained model weights
Internal research code and experimental algorithms
Customer-specific workflows and deployment infrastructure
Confidential performance benchmarks derived from private data
The included code is representative of the overall system design but does not contain all components used in production environments.

## Workflow

```bash

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

```

This repository is a public, redaction‑safe sample of a company program that employs an image recognition model to analyze plant root images. It demonstrates sample code without exposing any proprietary logic.

✅ You can share this repo publicly. Proprietary parsing rules, vendor/OCR config, and corp data are not included.

## Pipeline

```bash

The preprocessing workflow:

1. Load sample root images.
2. Apply multiple binarization methods:
   - Adaptive Thresholding
   - Frangi Vessel Enhancement
   - Morphological Segmentation
3. Compare generated masks against ground-truth masks.
4. Select the best-performing method.
5. Generate target maps for the full dataset.

```

## Model

The model performs multi-task learning:

### Root Weight Regression
Predicts total root biomass.

Loss:
- SmoothL1Loss

### Root Structure Prediction
Predicts dense root maps used as auxiliary supervision.

Loss:
- DenseThinRootHuberLoss

Combined Loss:

Loss = WeightLoss + 0.05 × MapLoss

## Quickstart

```bash

Clone the repository:

git clone https://github.com/wmeikle33/Greenscale.git
cd Greenscale
pip install -e .
pytest
python scripts/generate_demo_data.py
python -m image_analysis.pipeline
python -m image_analysis.train



```


### File Structure

<pre>
.
├── src/
├── tests/
├── docs/
├── LICENSE
├── README.md
