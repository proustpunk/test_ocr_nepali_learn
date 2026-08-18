# OCR Baseline Investigation — Devanagari

## Overview

This project establishes a baseline for Devanagari OCR before moving toward Vision-Language Model (VLM) evaluation and fine-tuning.

The initial objective is simple:

> **Measure how a conventional OCR system performs as the visual/textual complexity of the input increases.**

Tesseract OCR was evaluated against the ground-truth transcriptions provided by the HimalayaAI synthetic OCR benchmark.

The experiment currently focuses on five levels:

- `word`
- `sentence`
- `paragraph`
- `page`
- `exact`

The results are measured using **Character Error Rate (CER)**.

---

## Dataset

The experiment uses:

`himalaya-ai/nepalipixel-synthetic-ocr-benchmark`

The dataset provides:

- Devanagari images
- Ground-truth transcriptions
- A `level` field describing the type/complexity of the sample

For this experiment, **50 samples were randomly selected from each level** using a fixed random seed (`42`).

Total samples evaluated:

```text
250
```

---

## Experiment

### OCR Model

The baseline uses:

**Tesseract OCR**

with the Nepali language model:

```text
lang="nep"
```

Page segmentation mode was adjusted according to the sample type.

For word-level images:

```text
--psm 8
```

This tells Tesseract to treat the image as a single word.

For other levels:

```text
--psm 3
```

This uses Tesseract's automatic page segmentation.

This distinction is important because using the same segmentation configuration for a single isolated word and a full page does not represent the two recognition problems equally.

---

## Evaluation

The prediction from Tesseract is compared against the dataset's ground-truth transcription.

The primary metric is **Character Error Rate (CER)**.

CER measures the number of character-level edits required to transform the OCR prediction into the ground truth.

Lower is better.

The experiment records both:

1. Per-sample CER
2. Aggregate CER for each level

---

## Results

Current Tesseract results:

| Level | Samples | CER |
|---|---:|---:|
| Word | 50 | **0.0096** |
| Sentence | 50 | **0.0631** |
| Paragraph | 50 | **0.1063** |
| Page | 50 | **0.1193** |
| Exact | 50 | **0.1439** |

### Observation

The results reproduce the expected degradation in recognition performance as the input becomes more complex.

In particular:

```text
Word       → 0.0096
Sentence   → 0.0631
Paragraph  → 0.1063
Page       → 0.1193
```

Tesseract performs extremely well on isolated word recognition in this benchmark, while page-level recognition has substantially higher character error.

This provides an initial empirical baseline for investigating whether the same behavior occurs with VLM-based OCR.

---

## Important Caveat

This benchmark is synthetic, so these results should not be interpreted as a complete measure of real-world OCR robustness.

Real documents can introduce additional difficulties such as:

- blur
- noise
- uneven illumination
- perspective distortion
- compression artifacts
- complex backgrounds
- unusual fonts
- document layout
- tables and columns
- degraded scans
- mixed scripts and numerals

A future evaluation can therefore include real-world document images to test whether the observed degradation becomes more pronounced under realistic conditions.

---

## Project Structure

```text
root/
│
├── src/
│   └── ocr_test.py
│
├── dataset/
│   ├── images/
│   │   └── ...
│   └── metadata.json
│
├── results/
│   └── tesseract/
│       ├── predictions.json
│       └── cer_report.txt
│
└── README.md
```

### `dataset/`

Contains the selected benchmark samples and their metadata.

### `results/tesseract/`

Contains:

- `predictions.json` — ground truth, OCR prediction, level, PSM configuration, and per-sample CER
- `cer_report.txt` — aggregate CER by level

---

## Running the Experiment

Install the required dependencies:

```bash
pip install pytesseract pillow datasets jiwer
```

Tesseract itself must also be installed and the Nepali language data (`nep.traineddata`) must be available.

Run:

```bash
python src/ocr_test.py
```

The script will:

1. Load the Hugging Face dataset
2. Group samples by `level`
3. Select 50 samples from each level
4. Save the images locally
5. Run Tesseract OCR
6. Calculate per-sample CER
7. Save predictions
8. Calculate aggregate CER per level
9. Generate a text report

---

## Next Step

The next stage is to establish a **VLM baseline**.

The same general evaluation approach will be used:

```text
Image
  ↓
VLM
  ↓
Generated transcription
  ↓
Ground Truth
  ↓
CER
```

Candidate VLMs include models such as **Qwen-VL** and **Gemma-based vision-language models**.

The purpose is not simply to determine which model has the lowest CER, but to investigate:

- whether a VLM improves page-level recognition
- which types of Devanagari characters or structures remain difficult
- whether failures differ from conventional OCR
- whether the visual encoder preserves the information required for accurate Devanagari recognition

This provides the baseline for subsequent investigation into **VLM fine-tuning / SFT and eventually an end-to-end document OCR pipeline**.

---

## Current Status

### Completed

- [x] Load Devanagari OCR benchmark
- [x] Create controlled samples by recognition level
- [x] Establish Tesseract baseline
- [x] Evaluate against ground truth
- [x] Calculate CER
- [x] Reproduce degradation from word-level to page-level recognition

### Next

- [ ] Establish Qwen/Gemma VLM baseline
- [ ] Evaluate VLM using the same benchmark
- [ ] Compare OCR and VLM failure modes
- [ ] Investigate fine-tuning/SFT requirements
- [ ] Evaluate robustness on more realistic document conditions
- [ ] Move toward an end-to-end document OCR pipeline