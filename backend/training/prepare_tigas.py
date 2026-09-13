from pathlib import Path
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE_DIR / "dataset_tigas"

AI_LIMIT_PER_GENERATOR = 500
REAL_LIMIT = 1000

GENERATORS = {
    "DALLE2": "DALLE2",
    "SDXL": "sd_xl",
    "GauGAN": "gaugan",
    "StarGAN": "stargan",
}


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

AI_DIR = OUTPUT_DIR / "ai"
REAL_DIR = OUTPUT_DIR / "real"

AI_DIR.mkdir(parents=True, exist_ok=True)
REAL_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 60)
print("SignalScope - TIGAS Dataset Preparation")
print("=" * 60)

print("\nTarget:")
print(f"DALL-E 2: {AI_LIMIT_PER_GENERATOR}")
print(f"SDXL:     {AI_LIMIT_PER_GENERATOR}")
print(f"GauGAN:   {AI_LIMIT_PER_GENERATOR}")
print(f"StarGAN:  {AI_LIMIT_PER_GENERATOR}")
print(f"Real:     {REAL_LIMIT}")


# ============================================================
# LOAD TIGAS ANNOTATIONS
# ============================================================

print("\nLoading TIGAS annotations...")

url = (
    "https://huggingface.co/datasets/"
    "H1merka/TIGAS_dataset/"
    "resolve/main/train/annotations01.csv"
)

response = requests.get(
    url,
    timeout=60
)

response.raise_for_status()

df = pd.read_csv(
    BytesIO(response.content)
)

print(
    f"Total annotation rows: {len(df)}"
)


# ============================================================
# NORMALIZE PATHS
# ============================================================

df["image_path"] = (
    df["image_path"]
    .astype(str)
    .str.replace("\\", "/", regex=False)
)


# ============================================================
# FIND GENERATOR
# ============================================================

def get_generator(path):

    parts = path.split("/")

    if len(parts) < 2:
        return None

    return parts[1]


df["generator"] = df["image_path"].apply(
    get_generator
)


# ============================================================
# PRINT AVAILABLE GENERATORS
# ============================================================

print("\nAvailable generators:")

print(
    df["generator"]
    .value_counts()
    .to_string()
)


# ============================================================
# DOWNLOAD IMAGE
# ============================================================

def download_image(image_path):

    image_url = (
        "https://huggingface.co/datasets/"
        "H1merka/TIGAS_dataset/"
        "resolve/main/train/"
        + image_path
    )

    try:

        r = requests.get(
            image_url,
            timeout=30
        )

        if r.status_code != 200:
            return None

        image = Image.open(
            BytesIO(r.content)
        ).convert("RGB")

        return image

    except Exception:

        return None


# ============================================================
# COLLECT AI IMAGES
# ============================================================

print("\nCollecting AI images...")


for generator_name, generator_folder in GENERATORS.items():

    print(
        f"\nProcessing {generator_name}..."
    )

    generator_rows = df[
        (df["generator"] == generator_folder)
        &
        (df["label"] == 1)
    ]

    count = 0

    for _, row in generator_rows.iterrows():

        if count >= AI_LIMIT_PER_GENERATOR:
            break

        image_path = row["image_path"]

        output_path = (
            AI_DIR
            /
            f"{generator_name.lower()}_{count:04d}.jpg"
        )

        if output_path.exists():

            count += 1
            continue

        image = download_image(
            image_path
        )

        if image is None:
            continue

        image.save(
            output_path,
            "JPEG",
            quality=95
        )

        count += 1

        print(
            f"\r{generator_name}: "
            f"{count}/{AI_LIMIT_PER_GENERATOR}",
            end=""
        )

    print()

    print(
        f"{generator_name}: "
        f"{count} images saved"
    )


# ============================================================
# COLLECT REAL IMAGES
# ============================================================

print("\nCollecting real images...")

real_rows = df[
    df["label"] == 0
]

real_count = 0

for _, row in real_rows.iterrows():

    if real_count >= REAL_LIMIT:
        break

    image_path = row["image_path"]

    output_path = (
        REAL_DIR
        /
        f"real_{real_count:04d}.jpg"
    )

    if output_path.exists():

        real_count += 1
        continue

    image = download_image(
        image_path
    )

    if image is None:
        continue

    image.save(
        output_path,
        "JPEG",
        quality=95
    )

    real_count += 1

    print(
        f"\rReal: "
        f"{real_count}/{REAL_LIMIT}",
        end=""
    )

print()

print(
    f"Real images saved: "
    f"{real_count}"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TIGAS PREPARATION COMPLETE")
print("=" * 60)

print(
    f"AI images: "
    f"{len(list(AI_DIR.glob('*.jpg')))}"
)

print(
    f"Real images: "
    f"{len(list(REAL_DIR.glob('*.jpg')))}"
)

print(
    f"\nDataset saved to:"
)

print(
    OUTPUT_DIR
)

print("=" * 60)