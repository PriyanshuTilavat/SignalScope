const API_URL = "http://127.0.0.1:8000";

export type BackendResult = {
  filename: string;
  message: string;
  status: string;

  // Prediction probabilities
  ai_probability: number;
  real_probability: number;

  // Percentages for UI
  ai_percent: number;
  real_percent: number;

  // Model information
  confidence: number;
  verdict: string;

  // Grad-CAM heatmap
  gradcam: string | null;
};


export async function analyzeImageFromBackend(
  file: File
): Promise<BackendResult> {

  const formData = new FormData();

  formData.append("file", file);


  const response = await fetch(
    `${API_URL}/api/predict`,
    {
      method: "POST",
      body: formData,
    }
  );


  if (!response.ok) {

    const errorText = await response.text();

    throw new Error(
      errorText || "Analysis failed."
    );
  }


  return response.json();
}