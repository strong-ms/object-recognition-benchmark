<div align="center">
  <h1>🎯 Object Recognition Benchmark</h1>
  <p>A comprehensive benchmarking pipeline for object detection and tracking across state-of-the-art AI models.</p>

</div>

<br />

## Overview

This project provides a benchmarking pipeline for object detection and tracking, comparing performance across multiple AI models and approaches.

Currently evaluated models:
- YOLO
- RT-DETR
- Florence-2-large (Microsoft)

---

## Repository Structure

```text
.
├── input/                  # Raw source video files to be processed
├── output/                 # Generated outputs
│   ├── bbox/               # Detection coordinates (e.g., detections.json)
│   ├── metrics/            # Frame-by-frame metric logs (.csv format)
│   └── video/              # Rendered videos overlaying predictions
├── scripts/                # Execution and inference scripts
│   ├── rtdetr.py           # RT-DETR inference & metrics logging
│   ├── yolo.py             # YOLO inference & metrics logging
│   ├── florence.py         # Florence-2 inference (phrase grounding to JSON)
│   └── florence_render.py  # Offline video rendering from Florence outputs
├── requirements.txt        # Python dependencies
└── README.md               # This documentation file
```

---

## Requirements

> [!CAUTION]  
> **This project requires GPU Acceleration**  
> To properly leverage NVIDIA Tensor Cores and avoid extreme CPU fallback delays, you must run this on a CUDA-enabled GPU.
> - Install PyTorch with the correct CUDA footprint matching your hardware.
> - Ensure your NVIDIA drivers are up to date.

---

## Installation

We recommend isolating the project dependencies using a Python virtual environment.

**Windows:**
```bash
python -m venv orb_env
orb_env\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv orb_env
source orb_env/bin/activate
```

### 2. Install Dependencies

Ensure you have activated the environment before installing:

```bash
pip install -r requirements.txt
```

---

## Usage

### YOLO Models Evaluation

```bash
python scripts/yolo.py
```

### RT-DETR Evaluation

```bash
python scripts/rtdetr.py
```

### Florence-2 Evaluation Pipeline

**1. Inference & Metrics (Logic Step):**

```bash
python scripts/florence.py
```

**2. Offline Rendering (Video Construction Step):**

```bash
python scripts/florence_render.py
```

---

## Notes

- Florence-2 uses *phrase grounding* and post-processes outputs via regex.
- Detection data is stored separately to bypass real-time rendering bottlenecks.
- Videos are reconstructed offline for consistency across models.
