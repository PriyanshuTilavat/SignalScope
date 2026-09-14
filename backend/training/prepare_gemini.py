from pathlib import Path
import random
import shutil

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    BASE_DIR
    / "kaggle_gemini"
    / "Nano Banana 2.0 Dataset"
    / "dataset"
)

OUTPUT_DIR = BASE_DIR / "dataset_gemini"

SEED = 42

TRAIN_AI = 500
VAL_AI = 100
TEST_AI = 100

random.seed(SEED)


# --------------------------------------------------
# Find images
# --------------------------------------------------

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

images = [
    p
    for p in SOURCE_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

print(f"Found Gemini images: {len(images)}")


# --------------------------------------------------
# Validate count
# --------------------------------------------------

required = TRAIN_AI + VAL_AI + TEST_AI

if len(images) < required:
    raise RuntimeError(
        f"Not enough Gemini images. "
        f"Found {len(images)}, need {required}."
    )


# --------------------------------------------------
# Shuffle
# --------------------------------------------------

random.shuffle(images)

train_images = images[:TRAIN_AI]

val_images = images[
    TRAIN_AI:
    TRAIN_AI + VAL_AI
]

test_images = images[
    TRAIN_AI + VAL_AI:
    TRAIN_AI + VAL_AI + TEST_AI
]


# --------------------------------------------------
# Create folders
# --------------------------------------------------

folders = [
    OUTPUT_DIR / "train" / "ai",
    OUTPUT_DIR / "train" / "real",
    OUTPUT_DIR / "val" / "ai",
    OUTPUT_DIR / "val" / "real",
    OUTPUT_DIR / "test" / "ai",
    OUTPUT_DIR / "test" / "real",
]

for folder in folders:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# --------------------------------------------------
# Copy Gemini images
# --------------------------------------------------

def copy_images(images, destination, prefix):
    for index, image_path in enumerate(images):
        destination_file = (
            destination
            / f"{prefix}_{index:05d}{image_path.suffix.lower()}"
        )

        shutil.copy2(
            image_path,
            destination_file
        )


print("Copying Gemini training images...")

copy_images(
    train_images,
    OUTPUT_DIR / "train" / "ai",
    "gemini_train"
)

print("Copying Gemini validation images...")

copy_images(
    val_images,
    OUTPUT_DIR / "val" / "ai",
    "gemini_val"
)

print("Copying Gemini test images...")

copy_images(
    test_images,
    OUTPUT_DIR / "test" / "ai",
    "gemini_test"
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("=" * 50)
print("GEMINI DATASET CREATED")
print("=" * 50)

print(f"Source images : {len(images)}")
print(f"Train AI      : {len(train_images)}")
print(f"Validation AI : {len(val_images)}")
print(f"Test AI       : {len(test_images)}")

print()
print(f"Output: {OUTPUT_DIR}")
print("=" * 50)