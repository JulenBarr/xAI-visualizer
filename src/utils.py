"""
Utility functions for image preprocessing and tensor conversions
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, List
from torchvision import transforms

from config import IMAGENET_MEAN, IMAGENET_STD, SUPPORTED_IMAGE_FORMATS, DEFAULT_INPUT_SIZES


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int],
    normalize: bool = True
) -> torch.Tensor:
    """
    Preprocess an image for model inference.

    Args:
        image: PIL Image to preprocess
        target_size: Target size as (height, width)
        normalize: Whether to apply ImageNet normalization

    Returns:
        Preprocessed image tensor of shape [1, 3, H, W]
    """
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Build transformation pipeline
    transform_list = [
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )

    transform = transforms.Compose(transform_list)

    # Apply transformations and add batch dimension
    tensor = transform(image).unsqueeze(0)

    return tensor


def tensor_to_image(tensor: torch.Tensor, denormalize: bool = False) -> np.ndarray:
    """
    Convert a tensor to a numpy array image.

    Args:
        tensor: Tensor of shape [C, H, W] or [1, C, H, W]
        denormalize: Whether to reverse ImageNet normalization

    Returns:
        Numpy array of shape [H, W, C] with values in range [0, 255]
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Denormalize if requested
    if denormalize:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = tensor * std + mean

    # Convert to numpy and transpose to [H, W, C]
    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    # Clip to valid range and convert to uint8
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)

    return image


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Validate that a filename has an allowed extension.

    Args:
        filename: Name of the file to validate
        allowed_extensions: List of allowed extensions (e.g., ['.jpg', '.png'])

    Returns:
        True if extension is valid, False otherwise
    """
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)


def get_model_input_size(model_type: str) -> Tuple[int, int]:
    """
    Get the default input size for a given model type.

    Args:
        model_type: Type of model ('ResNet' or 'YOLO')

    Returns:
        Tuple of (height, width)
    """
    return DEFAULT_INPUT_SIZES.get(model_type, (224, 224))


def normalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Normalize a tensor to range [0, 1] using min-max normalization.

    Args:
        tensor: Input tensor

    Returns:
        Normalized numpy array
    """
    array = tensor.detach().cpu().numpy()

    min_val = array.min()
    max_val = array.max()

    # Avoid division by zero
    if max_val - min_val < 1e-8:
        return np.zeros_like(array)

    normalized = (array - min_val) / (max_val - min_val)
    return normalized


def save_temp_file(uploaded_file, suffix: str) -> str:
    """
    Save an uploaded file to a temporary location.

    Args:
        uploaded_file: Streamlit UploadedFile object
        suffix: File suffix/extension

    Returns:
        Path to the saved temporary file
    """
    import tempfile
    import os

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    return temp_file.name
