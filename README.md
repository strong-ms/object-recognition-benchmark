<div align="center">
  <h1>Object Recognition Benchmark</h1>
  <p>A comprehensive benchmarking pipeline for object/action detection and tracking across state-of-the-art AI models.</p>

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
├── models/                 # Execution and inference scripts
│   ├── rtdetr/             # RT-DETR inference & metrics logging
│   ├── yolo/               # YOLO inference & metrics logging
│   ├── florence/           # Florence-2 inference (phrase grounding to JSON)
│   └── yowo/               # YOWO repository
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
python models/yolo/yolo.py
```

### RT-DETR Evaluation

```bash
python models/rtdetr/rtdetr.py
```

### Florence-2 Evaluation Pipeline

**1. Inference & Metrics (Logic Step):**

```bash
python models/florence2/florence.py
```

**2. Offline Rendering (Video Construction Step):**

```bash
python models/florence2/florence_render.py
```

### YOWO Evaluation

```bash
cd models/yowo
python test_video_ava.py --cfg cfg/ava.yaml
```

---

## Notes

- Florence-2 uses *phrase grounding* and post-processes outputs via regex.
- Detection data is stored separately to bypass real-time rendering bottlenecks.
- Videos are reconstructed offline for consistency across models.
