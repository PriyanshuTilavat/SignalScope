from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parents[1]

SOURCE = BASE_DIR / "dataset_expanded"
GEMINI = BASE_DIR / "dataset_gemini"
OUTPUT = BASE_DIR / "dataset_gemini_expanded"


def copy_folder(source, destination):
    destination.mkdir(parents=True, exist_ok=True)

    for file in source.iterdir():
        if file.is_file():
            shutil.copy2(file, destination / file.name)


# --------------------------------------------------
# Create output folders
# --------------------------------------------------

for split in ["train", "val"]:
    for label in ["ai", "real"]:
        (OUTPUT / split / label).mkdir(
            parents=True,
            exist_ok=True
        )


# --------------------------------------------------
# Copy existing expanded dataset
# --------------------------------------------------

print("Copying existing expanded training data...")

for split in ["train", "val"]:
    for label in ["ai", "real"]:
        source = SOURCE / split / label
        destination = OUTPUT / split / label

        copy_folder(source, destination)


# --------------------------------------------------
# Add Gemini training + validation images
# --------------------------------------------------

print("Adding Gemini training images...")

copy_folder(
    GEMINI / "train" / "ai",
    OUTPUT / "train" / "ai"
)

print("Adding Gemini validation images...")

copy_folder(
    GEMINI / "val" / "ai",
    OUTPUT / "val" / "ai"
)


# --------------------------------------------------
# Print counts
# --------------------------------------------------

print()
print("=" * 60)
print("GEMINI EXPANDED DATASET")
print("=" * 60)

for split in ["train", "val"]:
    for label in ["ai", "real"]:
        count = len(
            list(
                (OUTPUT / split / label).iterdir()
            )
        )

        print(
            f"{split:5} | "
            f"{label:4} | "
            f"{count} images"
        )

print()
print(f"Output: {OUTPUT}")
print("=" * 60)