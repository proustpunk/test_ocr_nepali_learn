import json
import os

import pytesseract
from datasets import load_dataset
from jiwer import cer
from PIL import Image


# ============================================================
# Configuration
# ============================================================

SAMPLES_PER_LEVEL = 50
RANDOM_SEED = 42


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.json")

RESULTS_DIR = os.path.join(BASE_DIR, "results", "tesseract")
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "predictions.json")
CER_REPORT_PATH = os.path.join(RESULTS_DIR, "cer_report.txt")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Load Hugging Face dataset
# ============================================================

print("Loading dataset...")

ds = load_dataset(
    "himalaya-ai/nepalipixel-synthetic-ocr-benchmark"
)

print(ds)
print("Columns:", ds["train"].column_names)


# ============================================================
# Group dataset by level
# ============================================================

train_data = ds["train"]

levels = {}

for sample in train_data:

    level = sample["level"]

    if level not in levels:
        levels[level] = []

    levels[level].append(sample)


print("\nAvailable levels:")

for level, samples in levels.items():
    print(f"  {level}: {len(samples)} samples")


# ============================================================
# Select samples
# ============================================================

selected_samples = []

for level, samples in levels.items():

    level_dataset = (
        ds["train"]
        .filter(lambda x: x["level"] == level)
        .shuffle(seed=RANDOM_SEED)
        .select(range(min(SAMPLES_PER_LEVEL, len(samples))))
    )

    selected_samples.extend(level_dataset)

    print(
        f"Selected {len(level_dataset)} samples "
        f"for level: {level}"
    )


print(
    f"\nTotal selected samples: "
    f"{len(selected_samples)}"
)


# ============================================================
# Save images and metadata
# ============================================================

metadata = []

print("\nSaving images...")

for i, sample in enumerate(selected_samples):

    image = sample["image"]
    ground_truth = sample["text"]
    level = sample["level"]

    image_filename = f"{i:04d}.png"
    image_path = os.path.join(
        IMAGE_DIR,
        image_filename
    )

    image.save(image_path)

    metadata.append({
        "id": i,
        "image": f"images/{image_filename}",
        "level": level,
        "ground_truth": ground_truth
    })


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2
    )


print(f"Metadata saved to: {METADATA_PATH}")


# ============================================================
# Run Tesseract + calculate per-sample CER
# ============================================================

print("\nRunning Tesseract...\n")

results = []

for sample in metadata:

    image_path = os.path.join(
        DATASET_DIR,
        sample["image"]
    )

    ground_truth = sample["ground_truth"]
    level = sample["level"]


    # --------------------------------------------------------
    # Choose page segmentation mode
    # --------------------------------------------------------

    if level == "word":
        config = "--psm 8"
    else:
        config = "--psm 3"


    # --------------------------------------------------------
    # Run OCR
    # --------------------------------------------------------

    with Image.open(image_path) as image:

        prediction = pytesseract.image_to_string(
            image,
            lang="nep",
            config=config
        ).strip()


    # --------------------------------------------------------
    # Calculate CER
    # --------------------------------------------------------

    sample_cer = cer(
        ground_truth,
        prediction
    )


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    result = {
        "id": sample["id"],
        "image": sample["image"],
        "level": level,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "cer": sample_cer,
        "psm": config
    }

    results.append(result)


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print("=" * 80)
    print(f"ID: {sample['id']}")
    print(f"Level: {level}")
    print(f"PSM: {config}")
    print(f"GT : {ground_truth}")
    print(f"OCR: {prediction}")
    print(f"CER: {sample_cer:.4f}")


# ============================================================
# Save predictions
# ============================================================

with open(
    PREDICTIONS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\nPredictions saved to:")
print(PREDICTIONS_PATH)


# ============================================================
# Calculate aggregate CER per level
# ============================================================

level_results = {}

for result in results:

    level = result["level"]

    if level not in level_results:

        level_results[level] = {
            "errors": 0,
            "characters": 0,
            "samples": 0
        }


    gt = result["ground_truth"]

    # CER × reference length gives the number of
    # character errors for that individual sample.

    level_results[level]["errors"] += (
        result["cer"] * len(gt)
    )

    level_results[level]["characters"] += len(gt)

    level_results[level]["samples"] += 1


# ============================================================
# Generate CER report
# ============================================================

report_lines = []

report_lines.append(
    "TESSERACT OCR - CER REPORT"
)

report_lines.append(
    "=" * 60
)

report_lines.append(
    f"Samples per level: {SAMPLES_PER_LEVEL}"
)

report_lines.append(
    f"Random seed: {RANDOM_SEED}"
)

report_lines.append("")


print("\n" + "=" * 60)
print("TESSERACT OCR - CER BY LEVEL")
print("=" * 60)


for level, data in level_results.items():

    overall_cer = (
        data["errors"] / data["characters"]
        if data["characters"] > 0
        else 0
    )

    line = (
        f"{level:<15} "
        f"Samples: {data['samples']:<4} "
        f"CER: {overall_cer:.4f}"
    )

    print(line)

    report_lines.append(line)


# ============================================================
# Save CER report
# ============================================================

with open(
    CER_REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report_lines)
    )


print("\nCER report saved to:")
print(CER_REPORT_PATH)