"""
Model loading utilities for ResNet and YOLO models in .pth and .onnx formats
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import os

from config import EXTRACTABLE_LAYER_TYPES


class ModelLoader:
    """
    Handles loading and validation of deep learning models.
    Supports ResNet and YOLO models in .pth and .onnx formats.
    """

    @staticmethod
    def detect_format(file_path: str) -> str:
        """
        Detect the file format based on extension.

        Args:
            file_path: Path to the model file

        Returns:
            'pth' or 'onnx'

        Raises:
            ValueError: If format is not supported
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.pth', '.pt']:
            return 'pth'
        elif ext == '.onnx':
            return 'onnx'
        else:
            raise ValueError(f"Unsupported model format: {ext}")

    @staticmethod
    def load_resnet_pth(file_path: str, architecture: str = 'resnet50') -> nn.Module:
        """
        Load a ResNet model from a .pth file.

        Args:
            file_path: Path to the .pth file
            architecture: ResNet architecture (default: resnet50)

        Returns:
            Loaded PyTorch model
        """
        from torchvision import models

        # Get the model architecture
        model_map = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }

        if architecture.lower() not in model_map:
            raise ValueError(f"Unsupported ResNet architecture: {architecture}")

        # Create model instance
        model = model_map[architecture.lower()](weights=None)

        # Load state dict
        state_dict = torch.load(file_path, map_location='cpu')

        # Handle different state dict formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model

    @staticmethod
    def load_yolo_pth(file_path: str) -> nn.Module:
        """
        Load a YOLO model from a .pth file.

        Args:
            file_path: Path to the .pth file

        Returns:
            Loaded PyTorch model

        Raises:
            ImportError: If ultralytics is not installed
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package is required for YOLO models. "
                "Install with: pip install ultralytics"
            )

        # Load YOLO model
        yolo = YOLO(file_path)

        # Extract the underlying PyTorch model
        model = yolo.model
        model.eval()

        return model

    @staticmethod
    def load_onnx_as_pytorch(file_path: str) -> nn.Module:
        """
        Load an ONNX model and convert it to PyTorch.

        Args:
            file_path: Path to the .onnx file

        Returns:
            Converted PyTorch model

        Raises:
            ImportError: If onnx or onnx2pytorch is not installed
        """
        try:
            import onnx
            from onnx2pytorch import ConvertModel
        except ImportError:
            raise ImportError(
                "onnx and onnx2pytorch are required for ONNX models. "
                "Install with: pip install onnx onnx2pytorch"
            )

        # Load ONNX model
        onnx_model = onnx.load(file_path)

        # Convert to PyTorch
        pytorch_model = ConvertModel(onnx_model)
        pytorch_model.eval()

        return pytorch_model

    @staticmethod
    def load_model(file_path: str, model_type: str, architecture: Optional[str] = None) -> nn.Module:
        """
        Main method to load a model based on file type and model type.

        Args:
            file_path: Path to the model file
            model_type: Type of model ('ResNet' or 'YOLO')
            architecture: Specific architecture (e.g., 'resnet50', optional)

        Returns:
            Loaded PyTorch model

        Raises:
            ValueError: If model type or format is unsupported
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        file_format = ModelLoader.detect_format(file_path)

        # Handle .pth files
        if file_format == 'pth':
            if model_type == 'ResNet':
                arch = architecture or 'resnet50'
                return ModelLoader.load_resnet_pth(file_path, arch)
            elif model_type == 'YOLO':
                return ModelLoader.load_yolo_pth(file_path)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        # Handle .onnx files
        elif file_format == 'onnx':
            return ModelLoader.load_onnx_as_pytorch(file_path)

        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    @staticmethod
    def get_layers_info(model: nn.Module) -> List[Dict]:
        """
        Extract information about all layers in the model.

        Args:
            model: PyTorch model

        Returns:
            List of dictionaries containing layer information
            Each dict has: 'name', 'type', 'index'
        """
        layers_info = []
        index = 0

        for name, module in model.named_modules():
            # Skip the root module
            if name == '':
                continue

            module_type = type(module).__name__

            # Check if this is an extractable layer type
            if module_type in EXTRACTABLE_LAYER_TYPES:
                layers_info.append({
                    'name': name,
                    'type': module_type,
                    'index': index,
                    'display_name': ModelLoader._format_layer_name(name, module_type)
                })
                index += 1

        return layers_info

    @staticmethod
    def _format_layer_name(name: str, module_type: str) -> str:
        """
        Format a layer name for display.

        Args:
            name: Original layer name
            module_type: Type of the module

        Returns:
            Formatted display name
        """
        # Replace dots with arrows for better readability
        display_name = name.replace('.', ' → ')

        # Add the module type in parentheses
        display_name = f"{display_name} ({module_type})"

        return display_name

    @staticmethod
    def validate_model(model: nn.Module) -> bool:
        """
        Validate that the model is properly loaded.

        Args:
            model: PyTorch model to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if model is in eval mode
            if model.training:
                model.eval()

            # Check if model has parameters
            params = list(model.parameters())
            if len(params) == 0:
                return False

            return True
        except Exception:
            return False
