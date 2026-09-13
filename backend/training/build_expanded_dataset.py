from pathlib import Path
import shutil
import random


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

ORIGINAL_DATASET = BASE_DIR / "dataset"

TIGAS_DATASET = BASE_DIR / "dataset_tigas"

EXPANDED_DATASET = BASE_DIR / "dataset_expanded"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TIGAS_VAL_RATIO = 0.20


random.seed(RANDOM_SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("SignalScope Expanded Dataset Builder")
print("=" * 60)


# ============================================================
# CHECK INPUTS
# ============================================================

if not ORIGINAL_DATASET.exists():

    raise FileNotFoundError(
        f"Original dataset not found:\n"
        f"{ORIGINAL_DATASET}"
    )


if not TIGAS_DATASET.exists():

    raise FileNotFoundError(
        f"TIGAS dataset not found:\n"
        f"{TIGAS_DATASET}"
    )


# ============================================================
# CREATE DIRECTORIES
# ============================================================

train_ai = (
    EXPANDED_DATASET
    / "train"
    / "ai"
)

train_real = (
    EXPANDED_DATASET
    / "train"
    / "real"
)

val_ai = (
    EXPANDED_DATASET
    / "val"
    / "ai"
)

val_real = (
    EXPANDED_DATASET
    / "val"
    / "real"
)


for directory in [
    train_ai,
    train_real,
    val_ai,
    val_real,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# COPY ORIGINAL DATASET
# ============================================================

print("\nCopying original training dataset...")


def copy_all(source, destination):

    files = list(
        source.glob("*")
    )

    count = 0

    for file in files:

        if not file.is_file():
            continue

        target = destination / file.name

        shutil.copy2(
            file,
            target
        )

        count += 1

    return count


original_train_ai = copy_all(
    ORIGINAL_DATASET / "train" / "ai",
    train_ai
)

original_train_real = copy_all(
    ORIGINAL_DATASET / "train" / "real",
    train_real
)


print(
    f"Original AI training images: "
    f"{original_train_ai}"
)

print(
    f"Original Real training images: "
    f"{original_train_real}"
)


print("\nCopying original validation dataset...")


original_val_ai = copy_all(
    ORIGINAL_DATASET / "val" / "ai",
    val_ai
)

original_val_real = copy_all(
    ORIGINAL_DATASET / "val" / "real",
    val_real
)


print(
    f"Original AI validation images: "
    f"{original_val_ai}"
)

print(
    f"Original Real validation images: "
    f"{original_val_real}"
)


# ============================================================
# SPLIT TIGAS AI
# ============================================================

print("\nSplitting TIGAS AI images...")


tigas_ai_files = list(
    (TIGAS_DATASET / "ai").glob("*.jpg")
)

random.shuffle(
    tigas_ai_files
)


ai_val_count = int(
    len(tigas_ai_files)
    * TIGAS_VAL_RATIO
)


tigas_ai_val = (
    tigas_ai_files[:ai_val_count]
)

tigas_ai_train = (
    tigas_ai_files[ai_val_count:]
)


# ============================================================
# COPY TIGAS AI
# ============================================================

for file in tigas_ai_train:

    shutil.copy2(
        file,
        train_ai / file.name
    )


for file in tigas_ai_val:

    shutil.copy2(
        file,
        val_ai / file.name
    )


print(
    f"TIGAS AI training: "
    f"{len(tigas_ai_train)}"
)

print(
    f"TIGAS AI validation: "
    f"{len(tigas_ai_val)}"
)


# ============================================================
# SPLIT TIGAS REAL
# ============================================================

print("\nSplitting TIGAS Real images...")


tigas_real_files = list(
    (TIGAS_DATASET / "real").glob("*.jpg")
)

random.shuffle(
    tigas_real_files
)


real_val_count = int(
    len(tigas_real_files)
    * TIGAS_VAL_RATIO
)


tigas_real_val = (
    tigas_real_files[:real_val_count]
)

tigas_real_train = (
    tigas_real_files[real_val_count:]
)


# ============================================================
# COPY TIGAS REAL
# ============================================================

for file in tigas_real_train:

    shutil.copy2(
        file,
        train_real / file.name
    )


for file in tigas_real_val:

    shutil.copy2(
        file,
        val_real / file.name
    )


print(
    f"TIGAS Real training: "
    f"{len(tigas_real_train)}"
)

print(
    f"TIGAS Real validation: "
    f"{len(tigas_real_val)}"
)


# ============================================================
# FINAL COUNTS
# ============================================================

final_train_ai = len(
    list(train_ai.glob("*.jpg"))
)

final_train_real = len(
    list(train_real.glob("*.jpg"))
)

final_val_ai = len(
    list(val_ai.glob("*.jpg"))
)

final_val_real = len(
    list(val_real.glob("*.jpg"))
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EXPANDED DATASET COMPLETE")
print("=" * 60)

print("\nTRAINING")

print(
    f"AI:   {final_train_ai}"
)

print(
    f"Real: {final_train_real}"
)

print(
    f"Total: {final_train_ai + final_train_real}"
)


print("\nVALIDATION")

print(
    f"AI:   {final_val_ai}"
)

print(
    f"Real: {final_val_real}"
)

print(
    f"Total: {final_val_ai + final_val_real}"
)


print("\nDataset location:")

print(
    EXPANDED_DATASET
)

print("=" * 60)