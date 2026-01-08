"""
Configuration constants for Explainable AI Visualization Tool
"""

# Supported file formats
SUPPORTED_MODEL_FORMATS = ['.pth', '.onnx', '.pt']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

# Model types
MODEL_TYPES = ['ResNet', 'YOLO', 'Faster R-CNN']

# Default input sizes for different model types
DEFAULT_INPUT_SIZES = {
    'ResNet': (224, 224),
    'YOLO': (640, 640),
    'Faster R-CNN': (800, 800)  # Can handle variable sizes, 800x800 is common
}

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Visualization settings
COLORMAP = 'jet'
OVERLAY_ALPHA = 0.4
GRID_MAX_COLS = 8
GRID_FIGSIZE_PER_COL = 2

# Layer types that can be visualized
EXTRACTABLE_LAYER_TYPES = [
    'Conv2d',
    'BatchNorm2d',
    'ReLU',
    'LeakyReLU',
    'MaxPool2d',
    'AvgPool2d',
    'AdaptiveAvgPool2d',
    'Bottleneck',
    'BasicBlock'
]

# Streamlit page configuration
PAGE_TITLE = "Explainable AI - Activation Visualizer"
PAGE_ICON = "🔍"
LAYOUT = "wide"
