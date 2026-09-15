import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from .preprocessing import preprocess_image


class GradCAM:
    """
    Grad-CAM implementation for SignalScope ResNet18.

    Generates a heatmap showing which image regions
    contributed most to the model's prediction.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.activations = None
        self.gradients = None

        # ResNet18 final convolutional layer
        self.target_layer = self.model.backbone.layer4[-1]

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )

        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image: Image.Image):
        """
        Generate a Grad-CAM heatmap.

        Returns:
            numpy array with values between 0 and 1.
        """

        self.model.eval()

        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.to(self.device)

        # Gradients must be enabled
        with torch.enable_grad():

            output = self.model(input_tensor)

            # Our model outputs REAL probability after sigmoid.
            real_probability = torch.sigmoid(output)

            # We want to explain the AI decision.
            ai_probability = 1.0 - real_probability

            # Clear previous gradients
            self.model.zero_grad()

            # Backpropagate AI probability
            ai_probability.backward()

        # Get saved feature maps
        activations = self.activations
        gradients = self.gradients

        if activations is None or gradients is None:
            raise RuntimeError(
                "Grad-CAM could not capture model activations."
            )

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted feature maps
        cam = (weights * activations).sum(
            dim=1,
            keepdim=True
        )

        # Remove negative values
        cam = F.relu(cam)

        # Resize to original image size
        cam = F.interpolate(
            cam,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        # Remove batch/channel dimensions
        cam = cam.squeeze().cpu().numpy()

        # Normalize between 0 and 1
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def close(self):
        """
        Remove PyTorch hooks.
        """

        self.forward_handle.remove()
        self.backward_handle.remove()