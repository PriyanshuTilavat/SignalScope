# SignalScope

## AI-Generated Image Detection System

SignalScope is a computer-vision system that estimates whether an uploaded image is likely AI-generated or likely real. It combines a trained ResNet18 classifier, calibrated-style probability reporting, Grad-CAM visual explanation, and a web interface.

> **Responsible-use note:** SignalScope provides a likelihood-based assessment. It should not be treated as definitive proof of image authenticity and should not be used for accusations or high-stakes decisions.

## Project Demonstration Video

Due to GitHub's file size limitations, the full video demonstration for this project is hosted on Google Drive.

You can view or download the video using the link below:

-> https://drive.google.com/file/d/1TFptErK_db63OZvosd1RYWaQ6Z2xEq2M/view?usp=drivesdk

---

## 1. Problem Statement

Modern image generators can create highly realistic images that are difficult to distinguish from photographs.

SignalScope addresses this problem by analyzing an image with a deep-learning classifier trained using real and AI-generated images from multiple generator families. The system reports:

- AI probability
- Real probability
- Confidence
- A likelihood-based verdict
- Grad-CAM visual explanation

The project also evaluates performance under generator and distribution shifts rather than reporting only an in-distribution validation score.

---

## 2. Main Features

- AI-generated vs real image classification
- AI probability
- Real probability
- Confidence score
- Likely AI-generated / Likely real verdict
- Grad-CAM visual explanation
- Image upload through a web interface
- Drag-and-drop upload
- Prediction history
- Clear history
- Multiple-generator evaluation
- Unseen-generator evaluation
- Robustness evaluation
- FastAPI backend
- React frontend
- CPU inference support

---

## 3. System Architecture

```text
                    Uploaded Image
                          |
                          v
                 Image Preprocessing
                          |
                          v
                    ResNet18 CNN
                          |
                          v
                Probability Estimation
                          |
              +-----------+-----------+
              |                       |
              v                       v
       AI / Real Verdict          Grad-CAM
              |                       |
              +-----------+-----------+
                          |
                          v
                    Web Interface
```

### Processing flow

1. User uploads an image.
2. The backend validates and loads the image.
3. The image is resized to 224 × 224 and normalized using ImageNet statistics.
4. The ResNet18 classifier produces a real-image probability.
5. AI probability is calculated as `1 - real_probability`.
6. The system reports the two probabilities, confidence, and a likelihood-based verdict.
7. Grad-CAM is generated from the final ResNet18 convolutional block.

---

## 4. Repository Structure

```text
SignalScope/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/
│   │   └── ml/
│   │       ├── model.py
│   │       ├── inference.py
│   │       ├── preprocessing.py
│   │       └── gradcam.py
│   ├── models/
│   │   └── signalscope_model.pth        # downloaded from Release
│   ├── training/
│   ├── evaluate_model.py
│   ├── robustness_test.py
│   └── requirements.txt
├── frontend/
│   ├── package.json
│   └── src/
├── report/
│   └── model_report.md
├── README.md
├── .gitignore
└── .gitattributes
```

> Large datasets and model weights are intentionally excluded from the normal Git history. The final model is distributed through the GitHub Release.

---

## 5. Model

### Final model

The final application uses the **Universal SignalScope model**, based on ResNet18.

- Backbone: ResNet18
- Framework: PyTorch
- Input size: 224 × 224
- Output: binary real/AI score
- Inference: CPU or CUDA when available
- Explainability: Grad-CAM
- Production threshold: 0.50 AI probability

The model's sigmoid output represents the **real-image probability**. The AI probability is the complement:

```text
AI probability = 1 - Real probability
```

The application uses likelihood wording:

```text
AI probability >= 50%  -> Likely AI-generated
AI probability < 50%   -> Likely real
```

---

## 6. Download the Final Model

The final Universal model weights are available from the GitHub Release:

**SignalScope v1.0.0 - Final Model**

https://github.com/PriyanshuTilavat/SignalScope/releases/tag/v1.0.0

Download:

```text
signalscope_model_universal.pth
```

Place it at:

```text
backend/models/signalscope_model_universal.pth
```

Then create the production filename:

```text
backend/models/signalscope_model.pth
```

For Windows PowerShell:

```powershell
cd E:\SignalScope\backend
Copy-Item models\signalscope_model_universal.pth models\signalscope_model.pth -Force
```

The repository intentionally does not commit `.pth` weights directly.

---

## 7. Dataset Sources and Attribution

The project used publicly available datasets for training and evaluation.

### Tiny-GenImage

Source:

https://huggingface.co/datasets/TheKernel01/Tiny-GenImage

Used for real/AI image training and validation.

### TIGAS

Source:

https://huggingface.co/datasets/H1merka/TIGAS_dataset

Used to expand generator diversity.

### AIGC Detection Benchmark

Source:

https://huggingface.co/datasets/TheKernel01/AIGC-Detection-Benchmark

Used for external generator/distribution-shift evaluation.

### Gemini / Nano Banana dataset

Dataset:

`ahnuf05/nano-banana-2-0-the-omni-subject-dataset`

Used to evaluate and improve behavior on Gemini-generated images.

Dataset licenses and original terms should be checked at their respective source pages before redistribution of the underlying images.

---

## 8. Data Split

For the Universal training experiment, the balanced dataset contained:

```text
Training:
  AI   = 3300
  Real = 3300

Validation:
  AI   = 700
  Real = 700
```

Gemini images were included in the Universal training/validation construction, while the dedicated Gemini test images used for the reported generator evaluation were kept separate from training and validation.

The project also evaluates an **unseen generator** separately to measure distribution-shift behavior.

The official organizer-held-out test set was not available during development, so the reported external/unseen results should not be interpreted as official organizer-test performance.

---

## 9. Reported Results

### Internal validation / model comparison

The previously expanded production baseline achieved:

| Metric   | Result |
| -------- | -----: |
| Accuracy | 87.00% |
| Macro-F1 |  ~0.87 |
| ROC-AUC  | 0.9438 |

The final Universal model was selected because it provided substantially better coverage of Gemini while retaining useful performance on other generator families and an unseen-generator evaluation.

### Final Universal generator evaluation

| Source / generator | Accuracy |
| ------------------ | -------: |
| Gemini             |      88% |
| DALL-E 2           |      81% |
| SDXL               |      91% |
| Real images        |      72% |
| CycleGAN (unseen)  |      41% |

### Unseen-generator evaluation

CycleGAN was excluded from the Universal training dataset and was used as an unseen-generator evaluation.

Combined CycleGAN + Real ROC-AUC:

```text
0.6057
```

This result demonstrates that generator generalization remains difficult and is an explicit limitation of the system.

### Confusion matrix

For the expanded internal validation model used as the stronger baseline:

```text
                Predicted AI    Predicted Real
Actual AI            776             124
Actual Real           84             616
```

Macro-F1 was approximately 0.87 and ROC-AUC was 0.9438.

The Universal model was selected for the application because its generator coverage was more aligned with the target use case, particularly Gemini.

---

## 10. Robustness Evaluation

A 200-image evaluation subset was used to test common image transformations.

Results for the baseline evaluation:

| Condition          | Accuracy | Macro-F1 |
| ------------------ | -------: | -------: |
| Original           |   85.00% |   85.58% |
| JPEG quality 40    |   82.00% |   82.18% |
| Gaussian blur      |   66.00% |   73.85% |
| Brightness 0.75    |   82.50% |   82.76% |
| Resize and restore |   84.50% |   85.84% |

The most significant observed weakness was Gaussian blur.

---

## 11. Explainability — Grad-CAM

SignalScope includes Grad-CAM to provide a visual explanation of image regions that influenced the classifier.

Target layer:

```text
ResNet18 -> layer4[-1]
```

The generated heatmap is overlaid on the uploaded image.

### Important limitation

Grad-CAM is an explanation of model activation, not proof that a particular object or region was generated by AI. Heatmaps should therefore be interpreted as supporting visual evidence rather than a forensic conclusion.

---

## 12. Responsible AI

SignalScope intentionally uses likelihood-based language.

It reports:

```text
Likely AI-generated
```

or:

```text
Likely real
```

rather than making absolute claims.

Known limitations include:

- Performance can change across unseen generators.
- Distribution shifts can reduce detection performance.
- Image compression and blur can affect predictions.
- A high confidence score does not guarantee correctness.
- Grad-CAM does not prove image provenance.
- The system should not be used for accusations or high-stakes decisions without independent evidence.

---

## 13. Setup and Run

### Requirements

- Python 3.x
- Node.js and npm
- PyTorch
- FastAPI
- Uvicorn
- React / Vite
- CPU is sufficient for inference

### Backend

From the project root:

```powershell
cd E:\SignalScope\backend
```

Create/activate the virtual environment as appropriate.

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Set the Python path:

```powershell
$env:PYTHONPATH="."
```

Start FastAPI:

```powershell
uvicorn app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

### Frontend

Open a second terminal:

```powershell
cd E:\SignalScope\frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally:

```text
http://localhost:5173
```

---

## 14. Prediction API

Endpoint:

```text
POST /api/predict
```

The endpoint accepts an uploaded image and returns prediction information including:

- AI probability
- Real probability
- Confidence
- Verdict
- Grad-CAM result

Example API flow:

```text
Frontend
   |
   | multipart image upload
   v
POST /api/predict
   |
   v
Preprocessing -> ResNet18 -> probabilities
   |
   +-> verdict
   |
   +-> Grad-CAM
   |
   v
JSON response
```

---

## 15. Reproducibility

Important training and evaluation scripts are included under:

```text
backend/training/
backend/
```

Examples include:

```text
train.py
train_expanded.py
train_gemini.py
build_universal_dataset.py
finetune-universal.py
evaluate_universal_auc.py
robustness_test.py
test_gradcam.py
```

Exact dataset contents are not committed to the public repository.

The final model weights are provided through the GitHub Release.

---

## 16. Demo

The final demonstration should show:

1. Starting the backend.
2. Starting the frontend.
3. Uploading a new AI-generated image.
4. Showing AI probability, real probability, confidence, and verdict.
5. Showing the Grad-CAM explanation.
6. Uploading a real image.
7. Showing prediction history and Clear History.
8. Briefly explaining evaluation results and known limitations.

Target demo duration:

```text
3–5 minutes
```

---

## 17. Project Status

**Status: Final prototype / SIH submission candidate**

Completed:

- [x] Real vs AI image classification
- [x] Multi-generator training/evaluation
- [x] Gemini evaluation
- [x] Unseen-generator evaluation
- [x] ROC-AUC evaluation
- [x] Macro-F1 evaluation
- [x] Confusion matrix evaluation
- [x] Robustness evaluation
- [x] Grad-CAM
- [x] FastAPI backend
- [x] React frontend
- [x] Prediction history
- [x] Clear history
- [x] Public GitHub repository
- [x] Final model release
- [x] Model report
- [ ] Final 3–5 minute demo recording

---

## 18. Model Report

The one-page model report is available at:

```text
report/model_report.md
```

It documents:

- Task
- Data and split
- Model and approach
- Training configuration
- Metrics
- Baseline
- Unseen-generator evaluation
- Robustness
- Grad-CAM
- Limitations
- Responsible-use considerations

---

## 19. Originality Declaration

This project was developed as a team implementation for SIH 2026.

Open-source libraries, public datasets, pretrained architectures, and AI coding assistants were used as development resources. The project team implemented the application, training/evaluation pipeline, model selection experiments, API, frontend, Grad-CAM integration, robustness evaluation, and documentation.

No public real-vs-AI detection notebook or complete third-party solution was copied wholesale as the submitted system.

The final system and its evaluation results are the responsibility of the project team.

---

## 20. Team / Submission Notes

SignalScope is intended as a research/prototype demonstration of AI-generated image detection.

The most important result is not a claim of perfect detection. The project explicitly measures performance across multiple generators and an unseen-generator setting and documents where the detector fails.

For the official organizer-held-out test, the organizer's test protocol and data should be treated as the authoritative evaluation.

---

## License / Third-Party Attribution

Third-party libraries and datasets remain subject to their original licenses and terms.

Please refer to the original dataset and library documentation before redistributing third-party data.
