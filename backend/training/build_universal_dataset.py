from pathlib import Path
import random
import shutil


# ============================================================
# SignalScope Universal Dataset Builder
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

# Existing expanded dataset
EXPANDED_DIR = BASE_DIR / "dataset_expanded"

# Gemini dataset
GEMINI_DIR = BASE_DIR / "dataset_gemini"

# New balanced universal dataset
OUTPUT_DIR = BASE_DIR / "dataset_universal"

# Reproducibility
SEED = 42
random.seed(SEED)


# ============================================================
# Configuration
# ============================================================

# Training
AI_TRAIN_TOTAL = 3300
REAL_TRAIN_TOTAL = 3300

# Validation
AI_VAL_TOTAL = 700
REAL_VAL_TOTAL = 700

# Gemini contribution
GEMINI_TRAIN_COUNT = 500
GEMINI_VAL_COUNT = 100


# ============================================================
# Supported image extensions
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# Utility: find images recursively
# ============================================================

def get_images(folder):
    if not folder.exists():
        raise FileNotFoundError(
            f"Folder does not exist:\n{folder}"
        )

    return [
        path
        for path in folder.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]


# ============================================================
# Utility: copy images
# ============================================================

def copy_images(
    images,
    destination,
    prefix
):
    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, source in enumerate(images):

        target = (
            destination
            / f"{prefix}_{index:05d}{source.suffix.lower()}"
        )

        shutil.copy2(
            source,
            target
        )


# ============================================================
# Validate source datasets
# ============================================================

print()
print("=" * 70)
print("SIGNALSCOPE UNIVERSAL DATASET BUILDER")
print("=" * 70)

print()
print("Checking source datasets...")


# Existing AI
existing_ai_train = get_images(
    EXPANDED_DIR / "train" / "ai"
)

existing_ai_val = get_images(
    EXPANDED_DIR / "val" / "ai"
)


# Existing real
existing_real_train = get_images(
    EXPANDED_DIR / "train" / "real"
)

existing_real_val = get_images(
    EXPANDED_DIR / "val" / "real"
)


# Gemini
gemini_train = get_images(
    GEMINI_DIR / "train" / "ai"
)

gemini_val = get_images(
    GEMINI_DIR / "val" / "ai"
)


# Gemini test is intentionally NOT used
gemini_test = get_images(
    GEMINI_DIR / "test" / "ai"
)


print()
print("SOURCE DATA")
print("-" * 70)

print(
    f"Existing AI train     : "
    f"{len(existing_ai_train)}"
)

print(
    f"Existing AI validation : "
    f"{len(existing_ai_val)}"
)

print(
    f"Existing real train   : "
    f"{len(existing_real_train)}"
)

print(
    f"Existing real val     : "
    f"{len(existing_real_val)}"
)

print(
    f"Gemini train          : "
    f"{len(gemini_train)}"
)

print(
    f"Gemini validation     : "
    f"{len(gemini_val)}"
)

print(
    f"Gemini test           : "
    f"{len(gemini_test)}"
)


# ============================================================
# Validate enough images exist
# ============================================================

existing_ai_train_needed = (
    AI_TRAIN_TOTAL
    - GEMINI_TRAIN_COUNT
)

existing_ai_val_needed = (
    AI_VAL_TOTAL
    - GEMINI_VAL_COUNT
)


if len(gemini_train) < GEMINI_TRAIN_COUNT:
    raise RuntimeError(
        f"Not enough Gemini training images.\n"
        f"Found: {len(gemini_train)}\n"
        f"Required: {GEMINI_TRAIN_COUNT}"
    )


if len(gemini_val) < GEMINI_VAL_COUNT:
    raise RuntimeError(
        f"Not enough Gemini validation images.\n"
        f"Found: {len(gemini_val)}\n"
        f"Required: {GEMINI_VAL_COUNT}"
    )


if len(existing_ai_train) < existing_ai_train_needed:
    raise RuntimeError(
        f"Not enough existing AI training images.\n"
        f"Found: {len(existing_ai_train)}\n"
        f"Required: {existing_ai_train_needed}"
    )


if len(existing_ai_val) < existing_ai_val_needed:
    raise RuntimeError(
        f"Not enough existing AI validation images.\n"
        f"Found: {len(existing_ai_val)}\n"
        f"Required: {existing_ai_val_needed}"
    )


if len(existing_real_train) < REAL_TRAIN_TOTAL:
    raise RuntimeError(
        f"Not enough existing real training images.\n"
        f"Found: {len(existing_real_train)}\n"
        f"Required: {REAL_TRAIN_TOTAL}"
    )


if len(existing_real_val) < REAL_VAL_TOTAL:
    raise RuntimeError(
        f"Not enough existing real validation images.\n"
        f"Found: {len(existing_real_val)}\n"
        f"Required: {REAL_VAL_TOTAL}"
    )


# ============================================================
# Shuffle source images
# ============================================================

random.shuffle(existing_ai_train)
random.shuffle(existing_ai_val)

random.shuffle(existing_real_train)
random.shuffle(existing_real_val)

random.shuffle(gemini_train)
random.shuffle(gemini_val)


# ============================================================
# Select images
# ============================================================

print()
print("Selecting images...")


# ------------------------------------------------------------
# AI TRAIN
# ------------------------------------------------------------

selected_existing_ai_train = (
    existing_ai_train[
        :existing_ai_train_needed
    ]
)

selected_gemini_train = (
    gemini_train[
        :GEMINI_TRAIN_COUNT
    ]
)

selected_ai_train = (
    selected_existing_ai_train
    + selected_gemini_train
)


# ------------------------------------------------------------
# AI VALIDATION
# ------------------------------------------------------------

selected_existing_ai_val = (
    existing_ai_val[
        :existing_ai_val_needed
    ]
)

selected_gemini_val = (
    gemini_val[
        :GEMINI_VAL_COUNT
    ]
)

selected_ai_val = (
    selected_existing_ai_val
    + selected_gemini_val
)


# ------------------------------------------------------------
# REAL TRAIN
# ------------------------------------------------------------

selected_real_train = (
    existing_real_train[
        :REAL_TRAIN_TOTAL
    ]
)


# ------------------------------------------------------------
# REAL VALIDATION
# ------------------------------------------------------------

selected_real_val = (
    existing_real_val[
        :REAL_VAL_TOTAL
    ]
)


# Shuffle final groups
random.shuffle(selected_ai_train)
random.shuffle(selected_ai_val)

random.shuffle(selected_real_train)
random.shuffle(selected_real_val)


# ============================================================
# Delete previous universal dataset
# ============================================================

if OUTPUT_DIR.exists():

    print()
    print(
        "Removing previous universal dataset..."
    )

    shutil.rmtree(OUTPUT_DIR)


# ============================================================
# Create folders
# ============================================================

print()
print("Creating output folders...")


folders = [
    OUTPUT_DIR / "train" / "ai",
    OUTPUT_DIR / "train" / "real",

    OUTPUT_DIR / "val" / "ai",
    OUTPUT_DIR / "val" / "real",
]


for folder in folders:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Copy TRAIN data
# ============================================================

print()
print("Copying training data...")


copy_images(
    selected_existing_ai_train,
    OUTPUT_DIR / "train" / "ai",
    "existing_ai"
)


copy_images(
    selected_gemini_train,
    OUTPUT_DIR / "train" / "ai",
    "gemini"
)


copy_images(
    selected_real_train,
    OUTPUT_DIR / "train" / "real",
    "real"
)


# ============================================================
# Copy VALIDATION data
# ============================================================

print()
print("Copying validation data...")


copy_images(
    selected_existing_ai_val,
    OUTPUT_DIR / "val" / "ai",
    "existing_ai_val"
)


copy_images(
    selected_gemini_val,
    OUTPUT_DIR / "val" / "ai",
    "gemini_val"
)


copy_images(
    selected_real_val,
    OUTPUT_DIR / "val" / "real",
    "real_val"
)


# ============================================================
# Final verification
# ============================================================

print()
print("=" * 70)
print("UNIVERSAL DATASET CREATED")
print("=" * 70)


train_ai_count = len(
    get_images(
        OUTPUT_DIR / "train" / "ai"
    )
)

train_real_count = len(
    get_images(
        OUTPUT_DIR / "train" / "real"
    )
)

val_ai_count = len(
    get_images(
        OUTPUT_DIR / "val" / "ai"
    )
)

val_real_count = len(
    get_images(
        OUTPUT_DIR / "val" / "real"
    )
)


print()
print(
    f"Train AI       : "
    f"{train_ai_count}"
)

print(
    f"Train Real     : "
    f"{train_real_count}"
)

print(
    f"Validation AI  : "
    f"{val_ai_count}"
)

print(
    f"Validation Real: "
    f"{val_real_count}"
)


# ============================================================
# Gemini information
# ============================================================

print()
print("Gemini contribution")
print("-" * 70)

print(
    f"Gemini train images     : "
    f"{GEMINI_TRAIN_COUNT}"
)

print(
    f"Gemini validation images: "
    f"{GEMINI_VAL_COUNT}"
)

print(
    f"Gemini test images      : "
    f"{len(gemini_test)}"
)

print()
print(
    "Gemini test images were NOT copied "
    "into the training or validation dataset."
)


# ============================================================
# Final status
# ============================================================

print()
print("=" * 70)

if (
    train_ai_count == 3300
    and train_real_count == 3300
    and val_ai_count == 700
    and val_real_count == 700
):

    print(
        "STATUS: DATASET READY FOR TRAINING"
    )

else:

    print(
        "STATUS: DATASET COUNT ERROR"
    )

print("=" * 70)
print()