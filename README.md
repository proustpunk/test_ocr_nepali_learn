# OCR Baseline Investigation — Devanagari

## Overview

This project establishes a controlled baseline for **Devanagari OCR** and investigates how recognition performance changes as visual and textual complexity increases.

The current evaluation compares:

- **Tesseract OCR** — conventional OCR baseline
- **Qwen3-VL** — Vision-Language Model OCR baseline

Both systems are evaluated against the same ground-truth dataset using **Character Error Rate (CER)**.

The immediate objective is:

> Measure OCR performance across different levels of document complexity and identify where recognition quality begins to degrade.

---

## Dataset

The experiment uses the:

`himalaya-ai/nepalipixel-synthetic-ocr-benchmark`

The dataset provides:

- Devanagari images
- Ground-truth transcriptions
- A `level` field describing the type/complexity of each sample

The current evaluation focuses on five levels:

- `word`
- `sentence`
- `paragraph`
- `page`
- `exact`

For each level, **50 samples** are randomly selected using a fixed random seed:

```text
Samples per level: 50
Random seed: 42
Total samples: 250