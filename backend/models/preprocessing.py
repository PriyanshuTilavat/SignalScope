from PIL import Image
from torchvision import transforms


# Image size expected by our neural network
IMAGE_SIZE = 224


# Same preprocessing will be used during training and prediction
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def preprocess_image(image: Image.Image):
    """
    Convert a PIL image into a PyTorch tensor
    that can be given to the AI model.
    """

    # Make sure image has 3 color channels
    image = image.convert("RGB")

    # Apply resize, tensor conversion and normalization
    tensor = image_transform(image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor