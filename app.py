"""
Explainable AI - Activation Visualization
Streamlit application for visualizing deep learning model activations
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.model_loader import ModelLoader
from src.activation_extractor import ActivationExtractor
from src.visualizer import ActivationVisualizer
from src.utils import preprocess_image, tensor_to_image, validate_file_extension, get_model_input_size, save_temp_file
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, MODEL_TYPES,
    SUPPORTED_MODEL_FORMATS, SUPPORTED_IMAGE_FORMATS, COLORMAP
)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    if 'extractor' not in st.session_state:
        st.session_state.extractor = None
    if 'layers_info' not in st.session_state:
        st.session_state.layers_info = []
    if 'current_activations' not in st.session_state:
        st.session_state.current_activations = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'preprocessed_tensor' not in st.session_state:
        st.session_state.preprocessed_tensor = None
    if 'selected_layer' not in st.session_state:
        st.session_state.selected_layer = None


def load_model_from_upload(uploaded_file, model_type: str, architecture: str = None):
    """
    Load a model from uploaded file.

    Args:
        uploaded_file: Streamlit UploadedFile object
        model_type: Type of model ('ResNet' or 'YOLO')
        architecture: Specific architecture (e.g., 'resnet50')
    """
    try:
        with st.spinner("Loading model..."):
            # Save uploaded file to temp location
            suffix = os.path.splitext(uploaded_file.name)[1]
            temp_path = save_temp_file(uploaded_file, suffix)

            # Load the model
            model = ModelLoader.load_model(temp_path, model_type, architecture)

            # Validate model
            if not ModelLoader.validate_model(model):
                st.error("Model validation failed. Please check the model file.")
                return

            # Get layers info
            layers_info = ModelLoader.get_layers_info(model)

            if not layers_info:
                st.warning("No extractable layers found in the model.")
                return

            # Store in session state
            st.session_state.model = model
            st.session_state.model_type = model_type
            st.session_state.layers_info = layers_info
            st.session_state.extractor = ActivationExtractor(model)

            st.success(f"✅ Model loaded successfully! Found {len(layers_info)} extractable layers.")

            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")


def process_image(uploaded_image):
    """
    Process uploaded image for inference.

    Args:
        uploaded_image: Streamlit UploadedFile object
    """
    try:
        # Load image
        image = Image.open(uploaded_image)

        # Store original
        st.session_state.original_image = np.array(image)

        # Preprocess for model
        target_size = get_model_input_size(st.session_state.model_type)
        preprocessed = preprocess_image(image, target_size, normalize=True)
        st.session_state.preprocessed_tensor = preprocessed

        st.success("✅ Image loaded and preprocessed!")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")


def extract_and_visualize(layer_name: str, viz_mode: str):
    """
    Extract activations and create visualization.

    Args:
        layer_name: Name of the layer to visualize
        viz_mode: Visualization mode ('grid' or 'heatmap')
    """
    try:
        with st.spinner("Extracting activations..."):
            # Register hook for selected layer
            st.session_state.extractor.register_hooks([layer_name])

            # Extract activations
            activations = st.session_state.extractor.extract_activations(
                st.session_state.preprocessed_tensor
            )

            # Get activation for this layer
            activation = activations.get(layer_name)

            if activation is None:
                st.error("Failed to extract activation for this layer.")
                return

            # Store activation
            st.session_state.current_activations = {layer_name: activation}

            # Create visualization based on mode
            if viz_mode == "Feature Maps Grid":
                with st.spinner("Creating feature grid visualization..."):
                    fig = ActivationVisualizer.create_feature_grid(
                        activation,
                        max_cols=8,
                        cmap='viridis',
                        max_channels=64  # Limit to first 64 channels
                    )
                    st.pyplot(fig)

                    # Show statistics
                    stats = ActivationVisualizer.get_activation_statistics(activation)
                    st.write("**Activation Statistics:**")
                    st.json(stats)

            elif viz_mode == "Aggregated Heatmap":
                with st.spinner("Creating heatmap visualization..."):
                    overlay = ActivationVisualizer.create_aggregated_heatmap(
                        activation,
                        st.session_state.original_image,
                        aggregation='mean'
                    )
                    st.image(overlay, caption="Activation Heatmap Overlay", use_container_width=True)

                    # Show statistics
                    stats = ActivationVisualizer.get_activation_statistics(activation)
                    st.write("**Activation Statistics:**")
                    st.json(stats)

    except Exception as e:
        st.error(f"Error during visualization: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT
    )

    # Initialize session state
    initialize_session_state()

    # Header
    st.title(f"{PAGE_ICON} Explainable AI - Activation Visualization")
    st.markdown(
        "Visualize what your deep learning models see by exploring activation functions "
        "from intermediate layers."
    )

    # Sidebar - Model Upload
    with st.sidebar:
        st.header("1. Load Model")

        model_type = st.selectbox(
            "Model Type",
            MODEL_TYPES,
            help="Select the type of model you're uploading"
        )

        # Architecture selector for ResNet
        architecture = None
        if model_type == "ResNet":
            architecture = st.selectbox(
                "Architecture",
                ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                index=2,  # Default to resnet50
                help="Select the ResNet architecture"
            )

        model_file = st.file_uploader(
            "Upload Model File",
            type=['pth', 'pt', 'onnx'],
            help="Upload a .pth or .onnx model file"
        )

        if model_file is not None:
            # Validate file extension
            if not validate_file_extension(model_file.name, SUPPORTED_MODEL_FORMATS):
                st.error(f"Unsupported file format. Use: {', '.join(SUPPORTED_MODEL_FORMATS)}")
            else:
                if st.button("Load Model", type="primary"):
                    load_model_from_upload(model_file, model_type, architecture)

        # Display model info if loaded
        if st.session_state.model is not None:
            st.success(f"Model: {st.session_state.model_type}")
            st.info(f"Layers: {len(st.session_state.layers_info)}")

        st.divider()

        # Image Upload
        st.header("2. Upload Image")

        image_file = st.file_uploader(
            "Upload Image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image for inference"
        )

        if image_file is not None and st.session_state.model is not None:
            # Display thumbnail
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Process Image", type="primary"):
                process_image(image_file)

        elif image_file is not None and st.session_state.model is None:
            st.warning("Please load a model first!")

    # Main Content Area
    if st.session_state.model is None:
        st.info("👈 Start by uploading a model from the sidebar")

        # Instructions
        st.markdown("### How to use:")
        st.markdown("""
        1. **Upload a Model**: Select your model type (ResNet or YOLO) and upload a `.pth` or `.onnx` file
        2. **Upload an Image**: Choose an image to run inference on
        3. **Select a Layer**: Pick which layer's activations you want to visualize
        4. **Choose Visualization Mode**: View as a feature grid or aggregated heatmap
        """)

        st.markdown("### Supported Models:")
        st.markdown("""
        - **ResNet**: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
        - **YOLO**: YOLOv8, YOLOv5 (and other Ultralytics models)
        """)

    elif st.session_state.preprocessed_tensor is None:
        st.info("👈 Upload an image to begin visualization")

    else:
        # Layer selection and visualization
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Layer Selection")

            # Create layer options
            layer_options = {
                layer['display_name']: layer['name']
                for layer in st.session_state.layers_info
            }

            selected_display_name = st.selectbox(
                "Select Layer",
                options=list(layer_options.keys()),
                help="Choose which layer to visualize"
            )

            selected_layer_name = layer_options[selected_display_name]

            # Display layer info
            layer_info = next(
                (l for l in st.session_state.layers_info if l['name'] == selected_layer_name),
                None
            )

            if layer_info:
                st.write(f"**Type:** {layer_info['type']}")
                st.write(f"**Name:** {layer_info['name']}")

            st.divider()

            # Visualization mode
            st.subheader("Visualization Mode")

            viz_mode = st.radio(
                "Mode",
                ["Feature Maps Grid", "Aggregated Heatmap"],
                help="Choose how to visualize the activations"
            )

            if viz_mode == "Feature Maps Grid":
                st.caption("Displays individual activation channels as a grid of heatmaps")
            else:
                st.caption("Shows average activation overlaid on the original image")

            st.divider()

            # Visualize button
            if st.button("Visualize Activations", type="primary", use_container_width=True):
                extract_and_visualize(selected_layer_name, viz_mode)

        with col2:
            st.subheader("Activation Visualization")

            if st.session_state.current_activations is None:
                st.info("Select a layer and click 'Visualize Activations' to see the results")
            else:
                # Visualizations are displayed in extract_and_visualize()
                pass


if __name__ == "__main__":
    main()
