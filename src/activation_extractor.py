"""
Activation extraction using PyTorch forward hooks
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class ActivationExtractor:
    """
    Extracts intermediate layer activations from PyTorch models using forward hooks.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the activation extractor.

        Args:
            model: PyTorch model to extract activations from
        """
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List = []

    def _get_activation(self, name: str):
        """
        Create a hook function that captures layer output.

        Args:
            name: Name of the layer

        Returns:
            Hook function
        """
        def hook(module, input, output):
            # Detach to avoid keeping computation graph
            self.activations[name] = output.detach()

        return hook

    def register_hooks(self, layer_names: List[str]) -> None:
        """
        Register forward hooks on specified layers.

        Args:
            layer_names: List of layer names to register hooks on

        Raises:
            ValueError: If a layer name is not found in the model
        """
        # Clear any existing hooks first
        self.remove_hooks()

        # Get all modules in the model
        named_modules = dict(self.model.named_modules())

        # Register hooks for each specified layer
        for name in layer_names:
            if name not in named_modules:
                raise ValueError(f"Layer '{name}' not found in model")

            module = named_modules[name]
            hook = module.register_forward_hook(self._get_activation(name))
            self.hooks.append(hook)

    def remove_hooks(self) -> None:
        """
        Remove all registered hooks.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def extract_activations(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and extract activations from registered layers.

        Args:
            image_tensor: Input image tensor of shape [1, 3, H, W]

        Returns:
            Dictionary mapping layer names to activation tensors

        Raises:
            RuntimeError: If no hooks are registered
        """
        if not self.hooks:
            raise RuntimeError(
                "No hooks registered. Call register_hooks() first."
            )

        # Clear previous activations
        self.activations.clear()

        # Set model to eval mode
        self.model.eval()

        # Run inference without gradient computation
        with torch.no_grad():
            _ = self.model(image_tensor)

        return self.activations.copy()

    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        """
        Get activation for a specific layer.

        Args:
            layer_name: Name of the layer

        Returns:
            Activation tensor or None if not found
        """
        return self.activations.get(layer_name)

    def clear_activations(self) -> None:
        """
        Clear all stored activations to free memory.
        """
        self.activations.clear()

    def get_layer_output_shape(self, layer_name: str, input_shape: tuple) -> Optional[tuple]:
        """
        Get the output shape of a specific layer for given input shape.

        Args:
            layer_name: Name of the layer
            input_shape: Input tensor shape (e.g., (1, 3, 224, 224))

        Returns:
            Output shape tuple or None if layer not found
        """
        # Create dummy input
        dummy_input = torch.zeros(input_shape)

        # Register hook temporarily
        self.register_hooks([layer_name])

        # Run forward pass
        try:
            self.extract_activations(dummy_input)
            activation = self.get_activation(layer_name)

            if activation is not None:
                return tuple(activation.shape)
            return None
        except Exception:
            return None
        finally:
            self.remove_hooks()

    def __del__(self):
        """
        Cleanup hooks when the object is destroyed.
        """
        self.remove_hooks()
