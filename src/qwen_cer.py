import json
import os

from jiwer import cer


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "qwen"
)

PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "predictions_qwen.json"
)

CER_REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "cer_report_qwen.txt"
)


# ============================================================
# Load predictions
# ============================================================

with open(
    PREDICTIONS_PATH,
    "r",
    encoding="utf-8"
) as f:

    results = json.load(f)


# ============================================================
# Calculate per-sample CER
# ============================================================

for result in results:

    result["cer"] = cer(
        result["ground_truth"],
        result["prediction"]
    )


# ============================================================
# Aggregate CER by level
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

    level_results[level]["errors"] += (
        result["cer"] * len(gt)
    )

    level_results[level]["characters"] += len(gt)

    level_results[level]["samples"] += 1


# ============================================================
# Print per-sample results
# ============================================================

for result in results:

    print("=" * 80)
    print(f"ID: {result['id']}")
    print(f"Level: {result['level']}")
    print(f"GT   : {result['ground_truth']}")
    print(f"Qwen : {result['prediction']}")
    print(f"CER  : {result['cer']:.4f}")


# ============================================================
# Generate report
# ============================================================

report_lines = [
    "QWEN3-VL OCR - CER REPORT",
    "=" * 60,
    ""
]

print("\n" + "=" * 60)
print("QWEN3-VL OCR - CER BY LEVEL")
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
# Save updated predictions with CER
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


# ============================================================
# Save report
# ============================================================

with open(
    CER_REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(report_lines))


print("\nCER report saved to:")
print(CER_REPORT_PATH)