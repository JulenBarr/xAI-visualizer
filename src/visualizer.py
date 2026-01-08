"""
Visualization utilities for activation maps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Tuple
from matplotlib.figure import Figure

from config import COLORMAP, OVERLAY_ALPHA, GRID_MAX_COLS, GRID_FIGSIZE_PER_COL


class ActivationVisualizer:
    """
    Creates visualizations of neural network activations.
    """

    @staticmethod
    def create_feature_grid(
        activation: torch.Tensor,
        max_cols: int = GRID_MAX_COLS,
        cmap: str = 'viridis',
        max_channels: Optional[int] = None,
        channel_selection: str = 'first',
        channel_indices: Optional[list] = None
    ) -> Figure:
        """
        Create a grid visualization of feature maps.

        Args:
            activation: Activation tensor of shape [batch, channels, height, width]
            max_cols: Maximum number of columns in the grid
            cmap: Matplotlib colormap name
            max_channels: Maximum number of channels to display (None for all)
            channel_selection: 'first', 'top_activated', or 'custom'
            channel_indices: Custom list of channel indices (for 'custom' mode)

        Returns:
            Matplotlib figure containing the grid

        Raises:
            ValueError: If activation tensor has wrong shape
        """
        # Validate shape
        if activation.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor [batch, channels, height, width], "
                f"got {activation.dim()}D tensor"
            )

        # Extract channels from first batch
        batch, total_channels, height, width = activation.shape

        # Select channels based on method
        if channel_selection == 'top_activated':
            # Calculate mean activation per channel
            channel_means = activation[0].mean(dim=(1, 2))  # Average over H and W
            # Get indices of top activated channels
            _, top_indices = torch.topk(channel_means, min(max_channels or total_channels, total_channels))
            selected_indices = top_indices.tolist()
        elif channel_selection == 'custom' and channel_indices is not None:
            selected_indices = channel_indices[:max_channels] if max_channels else channel_indices
        else:  # 'first' or default
            selected_indices = list(range(min(max_channels or total_channels, total_channels)))

        # Select the channels
        activation = activation[:, selected_indices, :, :]
        channels = len(selected_indices)

        # Calculate grid dimensions
        n_cols = min(channels, max_cols)
        n_rows = (channels + n_cols - 1) // n_cols

        # Create figure
        figsize = (n_cols * GRID_FIGSIZE_PER_COL, n_rows * GRID_FIGSIZE_PER_COL)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Handle single subplot case
        if channels == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        axes = axes.flatten()

        # Plot each channel
        for idx in range(channels):
            channel_data = activation[0, idx].cpu().numpy()

            # Normalize channel data for better visualization
            channel_data = ActivationVisualizer.normalize_activation(
                torch.from_numpy(channel_data)
            )

            # Display the channel
            im = axes[idx].imshow(channel_data, cmap=cmap)
            axes[idx].axis('off')

            # Show actual channel index
            actual_channel_idx = selected_indices[idx]
            axes[idx].set_title(f'Ch {actual_channel_idx}', fontsize=10, fontweight='bold')

            # Add colorbar (smaller)
            plt.colorbar(im, ax=axes[idx], fraction=0.035, pad=0.02)

        # Hide unused subplots
        for idx in range(channels, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig

    @staticmethod
    def create_aggregated_heatmap(
        activation: torch.Tensor,
        original_image: np.ndarray,
        aggregation: str = 'mean',
        alpha: float = OVERLAY_ALPHA,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Create an aggregated heatmap overlaid on the original image.

        Args:
            activation: Activation tensor of shape [batch, channels, height, width]
            original_image: Original image as numpy array [H, W, C]
            aggregation: Aggregation method ('mean', 'max', or 'sum')
            alpha: Overlay transparency (0-1, lower = more original image visible)
            colormap: OpenCV colormap constant

        Returns:
            Overlaid image as numpy array [H, W, C]

        Raises:
            ValueError: If activation tensor has wrong shape or aggregation method invalid
        """
        # Validate shape
        if activation.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor [batch, channels, height, width], "
                f"got {activation.dim()}D tensor"
            )

        # Aggregate across channels
        if aggregation == 'mean':
            heatmap = torch.mean(activation[0], dim=0)
        elif aggregation == 'max':
            heatmap, _ = torch.max(activation[0], dim=0)
        elif aggregation == 'sum':
            heatmap = torch.sum(activation[0], dim=0)
        else:
            raise ValueError(
                f"Invalid aggregation method: {aggregation}. "
                f"Choose from 'mean', 'max', or 'sum'"
            )

        # Convert to numpy
        heatmap = heatmap.cpu().numpy()

        # Normalize to 0-255
        heatmap = ActivationVisualizer.normalize_activation(
            torch.from_numpy(heatmap)
        )
        heatmap = (heatmap * 255).astype(np.uint8)

        # Resize to match original image dimensions
        target_size = (original_image.shape[1], original_image.shape[0])
        heatmap_resized = cv2.resize(heatmap, target_size)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)

        # Ensure original image is in correct format
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)

        # Convert RGB to BGR if necessary (OpenCV uses BGR)
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        else:
            original_bgr = original_image

        # Overlay heatmap on original image
        overlay = cv2.addWeighted(
            original_bgr,
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )

        # Convert back to RGB for display
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        return overlay_rgb

    @staticmethod
    def normalize_activation(activation: torch.Tensor) -> np.ndarray:
        """
        Normalize activation using min-max normalization.

        Args:
            activation: Activation tensor

        Returns:
            Normalized numpy array in range [0, 1]
        """
        array = activation.detach().cpu().numpy()

        min_val = array.min()
        max_val = array.max()

        # Avoid division by zero
        if max_val - min_val < 1e-8:
            return np.zeros_like(array)

        normalized = (array - min_val) / (max_val - min_val)
        return normalized

    @staticmethod
    def get_activation_statistics(activation: torch.Tensor) -> dict:
        """
        Calculate statistics for an activation tensor.

        Args:
            activation: Activation tensor of shape [batch, channels, height, width]

        Returns:
            Dictionary containing statistics
        """
        stats = {
            'shape': tuple(activation.shape),
            'mean': float(activation.mean()),
            'std': float(activation.std()),
            'min': float(activation.min()),
            'max': float(activation.max()),
            'num_channels': activation.shape[1],
            'spatial_size': (activation.shape[2], activation.shape[3])
        }

        return stats

    @staticmethod
    def create_single_channel_view(
        activation: torch.Tensor,
        channel_idx: int,
        cmap: str = 'viridis'
    ) -> Figure:
        """
        Create a detailed view of a single activation channel.

        Args:
            activation: Activation tensor of shape [batch, channels, height, width]
            channel_idx: Index of the channel to visualize
            cmap: Matplotlib colormap name

        Returns:
            Matplotlib figure

        Raises:
            ValueError: If channel index is out of bounds
        """
        if channel_idx >= activation.shape[1]:
            raise ValueError(
                f"Channel index {channel_idx} out of bounds for "
                f"activation with {activation.shape[1]} channels"
            )

        # Extract channel
        channel_data = activation[0, channel_idx].cpu().numpy()

        # Normalize
        channel_data = ActivationVisualizer.normalize_activation(
            torch.from_numpy(channel_data)
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Display
        im = ax.imshow(channel_data, cmap=cmap)
        ax.set_title(f'Channel {channel_idx}', fontsize=12)
        ax.axis('off')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        return fig
