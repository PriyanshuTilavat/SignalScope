from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from app.ml.model import create_model


BASE_DIR = Path(__file__).resolve().parent

DATASET = BASE_DIR / "dataset_expanded" / "val"

MODELS = {
    "Production": BASE_DIR / "models" / "signalscope_model.pth",
    "Gemini Candidate": BASE_DIR / "models" / "signalscope_model_gemini.pth",
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model(path):
    model = create_model()

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


def evaluate(model):

    y_true = []
    y_pred = []
    y_score = []

    for label_name, label_value in [
        ("real", 0),
        ("ai", 1),
    ]:

        folder = DATASET / label_name

        for image_path in folder.iterdir():

            if image_path.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            ]:
                continue

            image = Image.open(
                image_path
            ).convert("RGB")

            tensor = transform(image)
            tensor = tensor.unsqueeze(0)
            tensor = tensor.to(device)

            with torch.no_grad():

                output = model(tensor)

                real_probability = (
                    torch.sigmoid(output).item()
                )

                ai_probability = (
                    1.0 - real_probability
                )

            prediction = (
                1 if ai_probability >= 0.5 else 0
            )

            y_true.append(label_value)
            y_pred.append(prediction)
            y_score.append(ai_probability)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    auc = roc_auc_score(
        y_true,
        y_score
    )

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    return accuracy, f1, auc, matrix


print()
print("=" * 70)
print("SIGNALSCOPE MODEL COMPARISON")
print("=" * 70)

print(f"Device: {device}")
print(f"Dataset: {DATASET}")

results = {}

for name, path in MODELS.items():

    print()
    print(f"Testing: {name}")

    model = load_model(path)

    accuracy, f1, auc, matrix = evaluate(model)

    results[name] = (
        accuracy,
        f1,
        auc,
        matrix
    )

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Macro-F1 : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("Confusion Matrix:")
    print(matrix)


print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

for name, values in results.items():

    accuracy, f1, auc, matrix = values

    print(
        f"{name:20} "
        f"Accuracy={accuracy * 100:.2f}% "
        f"F1={f1:.4f} "
        f"AUC={auc:.4f}"
    )