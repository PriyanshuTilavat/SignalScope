from pathlib import Path

from datasets import load_dataset
from PIL import Image


# ==========================================
# SignalScope Dataset Preparation
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"

TRAIN_AI = DATASET_DIR / "train" / "ai"
TRAIN_REAL = DATASET_DIR / "train" / "real"

VAL_AI = DATASET_DIR / "val" / "ai"
VAL_REAL = DATASET_DIR / "val" / "real"


# Create folders
for folder in [
    TRAIN_AI,
    TRAIN_REAL,
    VAL_AI,
    VAL_REAL,
]:
    folder.mkdir(parents=True, exist_ok=True)


TRAIN_PER_CLASS = 2500
VAL_PER_CLASS = 500


def save_split(split_name, ai_folder, real_folder, limit_per_class):
    print(f"\nLoading {split_name} split...")

    dataset = load_dataset(
        "TheKernel01/Tiny-GenImage",
        split=split_name,
        streaming=True,
    )

    ai_count = 0
    real_count = 0

    for item in dataset:

        label = item["label"]
        image = item["image"]

        # Tiny-GenImage:
        # 0 = real
        # 1 = fake / AI
        if label == 1 and ai_count < limit_per_class:
            image = image.convert("RGB")

            image.save(
                ai_folder / f"ai_{ai_count:05d}.jpg",
                "JPEG",
                quality=95,
            )

            ai_count += 1

        elif label == 0 and real_count < limit_per_class:
            image = image.convert("RGB")

            image.save(
                real_folder / f"real_{real_count:05d}.jpg",
                "JPEG",
                quality=95,
            )

            real_count += 1

        print(
            f"\rAI: {ai_count}/{limit_per_class} | "
            f"Real: {real_count}/{limit_per_class}",
            end="",
        )

        if (
            ai_count >= limit_per_class
            and real_count >= limit_per_class
        ):
            break

    print("\nFinished!")

    print(f"AI images:   {ai_count}")
    print(f"Real images: {real_count}")


# ==========================================
# TRAIN DATA
# ==========================================

save_split(
    "train",
    TRAIN_AI,
    TRAIN_REAL,
    TRAIN_PER_CLASS,
)


# ==========================================
# VALIDATION DATA
# ==========================================

save_split(
    "validation",
    VAL_AI,
    VAL_REAL,
    VAL_PER_CLASS,
)


print("\n==========================================")
print("SignalScope dataset preparation complete!")
print("==========================================")