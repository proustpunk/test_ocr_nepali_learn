import json
import os
import base64

from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.json")

RESULTS_DIR = os.path.join(BASE_DIR, "results", "qwen")
PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "predictions_qwen.json"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LM Studio
# ============================================================

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

MODEL_NAME = "qwen3-vl-4b-instruct"


# ============================================================
# Qwen OCR
# ============================================================

def qwen_ocr(image_path):

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe all text exactly as it appears "
                            "in the image. Do not translate, summarize, "
                            "explain, or correct anything. "
                            "Return only the transcription."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content.strip()


# ============================================================
# Load metadata
# ============================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


print(f"Loaded {len(metadata)} samples.")


# ============================================================
# Run Qwen
# ============================================================

results = []

for i, sample in enumerate(metadata, start=1):

    image_path = os.path.join(
        DATASET_DIR,
        sample["image"]
    )

    print(
        f"[{i}/{len(metadata)}] "
        f"ID: {sample['id']} "
        f"Level: {sample['level']}"
    )

    prediction = qwen_ocr(image_path)

    result = {
        "id": sample["id"],
        "image": sample["image"],
        "level": sample["level"],
        "ground_truth": sample["ground_truth"],
        "prediction": prediction
    }

    results.append(result)

    print(f"GT : {sample['ground_truth']}")
    print(f"Qwen: {prediction}")
    print()


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


print("=" * 60)
print("Qwen OCR complete.")
print(f"Predictions saved to:")
print(PREDICTIONS_PATH)