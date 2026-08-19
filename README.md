# Devanagari OCR Baseline: Tesseract vs Qwen3-VL

## Overview

This project establishes a baseline comparison between a conventional OCR system and a vision-language model (VLM) for Devanagari text recognition.

The purpose of the experiment is not to claim that either approach is universally better. Instead, it is an initial investigation into **where recognition performance degrades as the visual/textual complexity of an input increases**.

The current baseline compares:

* **Tesseract OCR**
* **Qwen3-VL**

Both are evaluated against the same ground-truth transcriptions from the HimalayaAI synthetic OCR benchmark.

The results provide a starting point for further investigation into VLM-based OCR, fine-tuning/SFT, and eventually more robust document-level OCR pipelines.

---

## Dataset

The experiment uses the Hugging Face dataset:

`himalaya-ai/nepalipixel-synthetic-ocr-benchmark`

The dataset contains:

* Devanagari images
* Ground-truth transcriptions
* A `level` field representing the type/complexity of the sample

The evaluated levels are:

* `word`
* `sentence`
* `paragraph`
* `page`
* `exact`

For each level, **50 samples** were randomly selected using a fixed random seed of `42`.

Total evaluation samples:

```text
250
```

The same selected samples are used for both Tesseract and Qwen3-VL so that the comparison is performed on the same ground truth.

---

# Experiment 1 — Tesseract OCR

## Configuration

Tesseract was used with the Nepali language model:

```text
lang="nep"
```

Page segmentation was adjusted for isolated words:

```text
word → --psm 8
other levels → --psm 3
```

`--psm 8` treats the input as a single word, while `--psm 3` uses automatic page segmentation.

This distinction is important because a single isolated word and a complete document page represent different OCR segmentation problems.

## Tesseract Results

| Level     | Samples |        CER |
| --------- | ------: | ---------: |
| Word      |      50 | **0.0096** |
| Sentence  |      50 | **0.0631** |
| Paragraph |      50 | **0.1063** |
| Page      |      50 | **0.1193** |
| Exact     |      50 | **0.1439** |

### Observation

Tesseract performs extremely well on isolated word recognition but its error rate increases as the input becomes more complex.

The important observation is the difference between:

```text
Word       0.0096
Sentence   0.0631
Paragraph  0.1063
Page       0.1193
```

This reproduces the expected behavior that **word-level recognition can be substantially easier than page-level recognition**.

However, this benchmark is synthetic and therefore does not fully represent the range of degradation encountered in real-world document images.

---

# Experiment 2 — Qwen3-VL

The next step was to evaluate a general-purpose VLM on the same benchmark.

The VLM receives the image and is prompted to produce the text contained in the image.

The resulting transcription is compared against the same ground truth using CER.

## Qwen3-VL Results

| Level     | Samples |        CER |
| --------- | ------: | ---------: |
| Word      |      50 | **0.1383** |
| Sentence  |      50 | **0.0951** |
| Paragraph |      50 | **0.0805** |
| Page      |      50 | **0.3647** |
| Exact     |      50 | **0.0678** |

### Observation

Qwen3-VL shows a substantially different error profile from Tesseract.

The most significant result is the page-level performance:

```text
Tesseract       Page CER: 0.1193
Qwen3-VL        Page CER: 0.3647
```

The VLM performs considerably worse on page-level transcription in this initial experiment.

At the same time, its behavior is not uniformly worse. For example:

```text
Qwen3-VL
Paragraph → 0.0805
Sentence  → 0.0951
Word      → 0.1383
```

This suggests that the problem is not simply:

> "VLMs are better than OCR."

or:

> "OCR is better than VLMs."

Instead, the two systems exhibit **different failure modes depending on the visual complexity and task format**.

---

# Comparison

| Level     | Tesseract CER | Qwen3-VL CER |
| --------- | ------------: | -----------: |
| Word      |    **0.0096** |       0.1383 |
| Sentence  |    **0.0631** |       0.0951 |
| Paragraph |        0.1063 |   **0.0805** |
| Page      |    **0.1193** |       0.3647 |
| Exact     |        0.1439 |   **0.0678** |

Lower CER is better.

The current results indicate:

* Tesseract is substantially better for isolated words.
* Qwen3-VL performs better on the paragraph and exact subsets in this particular evaluation.
* Both systems experience difficulties, but their degradation patterns are different.
* Qwen3-VL has a particularly large page-level degradation.
* A generic VLM therefore cannot simply be assumed to solve the document OCR problem.

---

# Why This Matters

The experiment raises a more useful question than simply choosing between OCR and VLM:

> **Why do both approaches fail, and what information is being lost at different stages of the pipeline?**

Conventional OCR systems are highly effective when the recognition problem is constrained. However, document images can introduce additional difficulties involving:

* text segmentation
* layout
* multiple lines
* columns
* background variation
* image degradation
* character/ligature recognition
* script-specific visual features

A VLM introduces a different architecture in which visual features are encoded and passed into a language model. However, a generic VLM is not necessarily optimized to preserve all of the fine-grained visual information required for accurate Devanagari OCR.

This motivates investigating the visual encoder, projector, language model interface, and fine-tuning process rather than assuming that simply adding a VLM will solve OCR.

---

# Current Findings

The initial experiments establish three useful observations.

### 1. Conventional OCR has a clear complexity-dependent failure pattern

Tesseract performs extremely well on isolated words but degrades on larger document structures.

### 2. A generic VLM does not automatically outperform conventional OCR

Qwen3-VL performs substantially worse than Tesseract on the page subset in this experiment.

### 3. The failure modes are not identical

The different CER profiles suggest that OCR and VLM systems are making different kinds of mistakes.

This makes direct comparison useful for identifying where a VLM-based OCR system needs improvement.

---

# Limitations

This experiment should be considered a **baseline investigation**, not a complete OCR benchmark.

### Synthetic data

The current evaluation is based on a synthetic OCR benchmark. It may not fully capture:

* camera photographs
* motion blur
* perspective distortion
* uneven lighting
* paper texture
* compression artifacts
* degraded scans
* unusual fonts
* complex real-world document layouts

Therefore, the current results should not be generalized directly to arbitrary real-world documents.

### Small evaluation set

Only 50 samples per level are currently evaluated.

This is sufficient for an initial baseline and failure-mode investigation, but not for statistically strong benchmarking.

### Model configuration

The Qwen3-VL experiment represents a baseline configuration. Different prompting, preprocessing, model sizes, decoding parameters, and fine-tuning strategies may produce different results.

---

# Project Structure

```text
root/
│
├── src/
│   ├── ocr_test.py
│   └── qwen_cer.py
│
├── dataset/
│   ├── images/
│   │   └── ...
│   └── metadata.json
│
├── results/
│   ├── tesseract/
│   │   ├── predictions.json
│   │   └── cer_report.txt
│   │
│   └── qwen/
│       ├── predictions_qwen.json
│       └── cer_report_qwen.txt
│
└── README.md
```

---

# Evaluation Pipeline

The current workflow is:

```text
Hugging Face Dataset
        │
        ▼
50 samples per level
        │
        ├──────────────────┐
        ▼                  ▼
   Tesseract            Qwen3-VL
        │                  │
        ▼                  ▼
  Transcription        Transcription
        │                  │
        └────────┬─────────┘
                 ▼
          Ground Truth
                 │
                 ▼
               CER
                 │
                 ▼
        Compare failure modes
```

---

# Running the Experiments

Install the Python dependencies:

```bash
pip install pytesseract pillow datasets jiwer
```

Tesseract itself must also be installed with the Nepali language data.

Run the Tesseract baseline:

```bash
python src/ocr_test.py
```

Run the Qwen3-VL evaluation and generate its CER report:

```bash
python src/qwen_cer.py
```

The exact inference script used to generate `predictions_qwen.json` can be added alongside the evaluation script.

---

# Next Steps

The next stage is to continue from the baseline rather than immediately assuming that fine-tuning is the solution.

## 1. Inspect VLM failures

Look at examples where Qwen3-VL produces high CER and determine what it is getting wrong.

Examples may include:

* individual Devanagari characters
* conjuncts/ligatures
* spacing
* line boundaries
* text ordering
* page-level structure
* visually ambiguous characters

## 2. Compare OCR and VLM failures

Determine whether the VLM fails on the same samples as Tesseract or whether the two systems fail differently.

## 3. Investigate VLM architecture

The next investigation can focus on how the visual encoder and projector represent Devanagari text before the information reaches the language model.

## 4. Fine-tuning / SFT

Once the baseline failure modes are understood, investigate whether supervised fine-tuning can improve the VLM's ability to perform Devanagari OCR.

The key question becomes:

> **What does the model need to learn that the generic VLM currently fails to preserve or recognize?**

## 5. Real-world evaluation

A later stage can introduce real-world document images alongside synthetic data to determine whether improvements transfer beyond the controlled benchmark.

---

# Status

### Completed

* [x] Load Devanagari OCR benchmark
* [x] Stratify samples by recognition level
* [x] Select 50 samples per level
* [x] Establish Tesseract baseline
* [x] Tune word-level Tesseract segmentation
* [x] Calculate per-sample CER
* [x] Calculate aggregate CER by level
* [x] Establish Qwen3-VL baseline
* [x] Compare Tesseract and Qwen3-VL CER
* [x] Identify substantially different page-level behavior

### Next

* [ ] Inspect high-CER Qwen3-VL examples
* [ ] Analyze OCR vs VLM failure modes
* [ ] Investigate visual encoder/projector behavior
* [ ] Evaluate VLM fine-tuning / SFT
* [ ] Test robustness on real-world documents
* [ ] Develop toward an end-to-end Devanagari document OCR pipeline

---

## Conclusion

The initial baseline demonstrates that **neither conventional OCR nor a generic VLM can simply be assumed to solve Devanagari document OCR**.

Tesseract provides very strong word-level recognition but exhibits increasing error with more complex inputs.

Qwen3-VL provides a different error profile and currently performs particularly poorly on the page-level subset.

The next phase is therefore not simply to find a model with a lower CER. It is to understand **why these failures occur and what needs to change in the VLM pipeline to make it reliable for Devanagari document recognition**.
