# Explainable AI - Activation Visualization Tool

A Streamlit-based web application for visualizing activation functions in deep learning models (ResNet and YOLO). This tool helps understand what patterns neural networks learn by displaying intermediate layer activations.

## Features

- **Model Support**: Load ResNet and YOLO models in `.pth` or `.onnx` formats
- **Interactive Layer Selection**: Choose any layer to visualize its activations
- **Two Visualization Modes**:
  - **Feature Maps Grid**: Display all activation channels as a grid of heatmaps
  - **Aggregated Heatmap**: Average across channels and overlay on the original image
- **Real-time Processing**: Upload images and see activations instantly
- **Detailed Statistics**: View activation statistics (mean, std, min, max, shape)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository:
```bash
cd c:\Users\julen\xAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser (typically at `http://localhost:8501`).

### Using the Tool

1. **Load a Model**:
   - Select model type (ResNet or YOLO)
   - For ResNet, choose the architecture (resnet18, resnet50, etc.)
   - Upload your model file (`.pth` or `.onnx`)
   - Click "Load Model"

2. **Upload an Image**:
   - Upload an image file (`.jpg`, `.png`, `.bmp`)
   - Click "Process Image"

3. **Visualize Activations**:
   - Select a layer from the dropdown
   - Choose visualization mode:
     - **Feature Maps Grid**: See individual activation channels
     - **Aggregated Heatmap**: See averaged activation overlay
   - Click "Visualize Activations"

## Supported Models

### ResNet
- ResNet18
- ResNet34
- ResNet50
- ResNet101
- ResNet152

### YOLO
- YOLOv5 (all variants)
- YOLOv8 (all variants)
- Other Ultralytics YOLO models

## File Formats

### Models
- `.pth` - PyTorch model files
- `.pt` - PyTorch model files
- `.onnx` - ONNX format (automatically converted to PyTorch)

### Images
- `.jpg`, `.jpeg` - JPEG images
- `.png` - PNG images
- `.bmp` - Bitmap images

## Project Structure

```
xAI/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration constants
├── README.md                   # This file
├── src/
│   ├── __init__.py
│   ├── model_loader.py         # Model loading utilities
│   ├── activation_extractor.py # Activation extraction via hooks
│   ├── visualizer.py           # Visualization functions
│   └── utils.py                # Helper utilities
├── tests/                      # Unit tests
└── assets/
    └── sample_images/          # Sample images for testing
```

## How It Works

### Activation Extraction

The tool uses PyTorch forward hooks to intercept and capture intermediate layer outputs:

1. **Hook Registration**: Hooks are registered on selected layers
2. **Forward Pass**: An image is passed through the model
3. **Activation Capture**: Layer outputs are captured and stored
4. **Visualization**: Activations are processed and displayed

### Visualization Modes

**Feature Maps Grid:**
- Displays up to 64 individual channels
- Each channel shown as a separate heatmap
- Useful for seeing specific feature detectors

**Aggregated Heatmap:**
- Averages all channels into a single heatmap
- Overlays on the original image
- Shows overall activation regions
- Helps identify which parts of the image activate the layer

## Examples

### Example 1: ResNet50 on an Image

```python
# Using a pretrained ResNet50
# 1. Download a pretrained model:
import torch
from torchvision import models

model = models.resnet50(pretrained=True)
torch.save(model.state_dict(), 'resnet50.pth')

# 2. Load in the app and visualize layer4.2.conv3
```

### Example 2: YOLO Model

```python
# Using a YOLO model
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Download from Ultralytics
# Load this file directly in the app
```

## Troubleshooting

### Model Loading Issues

**Problem**: "Model validation failed"
- **Solution**: Ensure the model file is not corrupted and matches the selected model type

**Problem**: ONNX conversion fails
- **Solution**: Try using the original PyTorch `.pth` format instead

### Visualization Issues

**Problem**: "No extractable layers found"
- **Solution**: The model may have an unusual architecture. Check that it contains standard layers (Conv2d, ReLU, etc.)

**Problem**: Out of memory errors
- **Solution**: Try visualizing layers with fewer channels, or use a smaller input image

### Installation Issues

**Problem**: CUDA/GPU errors
- **Solution**: The tool works on CPU by default. If you want GPU support, install the appropriate PyTorch version:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Dependencies

Core dependencies:
- `streamlit` - Web application framework
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision models and utilities
- `onnx` / `onnx2pytorch` - ONNX model support
- `ultralytics` - YOLO model support
- `opencv-python` - Image processing
- `matplotlib` - Visualization
- `numpy` - Numerical operations

See [requirements.txt](requirements.txt) for complete list with versions.

## Advanced Usage

### Custom Model Architectures

To add support for custom architectures, modify [src/model_loader.py](src/model_loader.py):

```python
# Add your custom model loading logic
def load_custom_model(file_path):
    model = YourCustomModel()
    model.load_state_dict(torch.load(file_path))
    return model
```

### Adding New Visualization Modes

Extend [src/visualizer.py](src/visualizer.py) to add new visualization types:

```python
@staticmethod
def create_custom_visualization(activation):
    # Your custom visualization logic
    pass
```

## Performance Tips

1. **Model Loading**: Models are cached in session state - you don't need to reload for each image
2. **Channel Limit**: Feature grid visualization is limited to 64 channels to prevent slowdown
3. **Image Size**: Larger images will take longer to process; consider resizing very large images

## Contributing

To contribute or extend this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open-source and available for educational and research purposes.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Uses [PyTorch](https://pytorch.org/) for model inference
- YOLO support via [Ultralytics](https://github.com/ultralytics/ultralytics)

## Future Enhancements

Potential features for future versions:
- Grad-CAM visualization
- Guided backpropagation
- Layer comparison view
- Batch processing
- Export visualizations as PDF reports
- Support for Vision Transformers
- Interactive layer architecture diagram

## Contact

For questions, issues, or suggestions, please open an issue on the project repository.
