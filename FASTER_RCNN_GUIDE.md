# Faster R-CNN Support Guide

## Overview

Your XAI Visualization Tool now supports **Faster R-CNN** models! Faster R-CNN is a popular object detection model that combines region proposals with deep convolutional networks.

## What is Faster R-CNN?

Faster R-CNN is a two-stage object detector:
1. **Region Proposal Network (RPN)**: Proposes candidate object regions
2. **Detection Network**: Classifies and refines bounding boxes

It's widely used for object detection tasks and understanding its activations helps you see:
- How the model identifies potential objects
- Which features contribute to detection
- How the backbone extracts visual features

## Supported Architectures

- **ResNet50 FPN** (Feature Pyramid Network) - Default and most common
- **ResNet50 FPN (standard)**
- **MobileNet V3** - Lighter, faster backbone

## Getting a Faster R-CNN Model

### Option 1: Use Pretrained Model (Easiest)

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load pretrained Faster R-CNN with ResNet50-FPN backbone
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Save it
torch.save(model.state_dict(), 'faster_rcnn_resnet50_fpn.pth')
print("Model saved as faster_rcnn_resnet50_fpn.pth")
```

### Option 2: Use MobileNet Backbone (Faster)

```python
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

# Load pretrained Faster R-CNN with MobileNet V3 backbone
model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

# Save it
torch.save(model.state_dict(), 'faster_rcnn_mobilenet_v3.pth')
print("Model saved!")
```

### Option 3: Use Your Custom-Trained Model

If you trained your own Faster R-CNN:

```python
# Your trained model
model = your_trained_faster_rcnn

# Save just the state dict
torch.save(model.state_dict(), 'my_faster_rcnn.pth')

# Or save with metadata
torch.save({
    'model': model.state_dict(),
    'architecture': 'resnet50_fpn',
    'classes': 91  # COCO classes
}, 'my_faster_rcnn_full.pth')
```

## Using Faster R-CNN in the App

### Step 1: Select Model Type

1. Open the app
2. In the sidebar, select **"Faster R-CNN"** from the Model Type dropdown
3. Choose backbone:
   - `resnet50` - Standard ResNet50 backbone
   - `resnet50_fpn` - ResNet50 with Feature Pyramid Network (recommended)
   - `mobilenet_v3` - MobileNet V3 backbone

### Step 2: Upload Model

1. Click "Upload Model File"
2. Select your `.pth` file
3. Click "Load Model"
4. Wait for model to load (may take a few seconds)

### Step 3: Upload Test Image

Good test images for object detection:
- Street scenes (cars, pedestrians, traffic signs)
- Indoor scenes (furniture, people, objects)
- Images with multiple objects

```python
# Download a test image
from PIL import Image
import requests
from io import BytesIO

url = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img.save('test_image.jpg')
```

### Step 4: Visualize Activations

**Interesting Layers to Explore:**

1. **Backbone Layers** (Feature Extraction):
   - `backbone.body.layer1` - Early features (edges, textures)
   - `backbone.body.layer2` - Mid-level features
   - `backbone.body.layer3` - High-level features
   - `backbone.body.layer4` - Semantic features

2. **FPN Layers** (Multi-scale Features):
   - `backbone.fpn.layer_blocks` - Feature pyramid levels
   - These combine features at different scales

3. **RPN Layers** (Region Proposals):
   - `rpn.head.conv` - Region proposal convolutions
   - Shows where the model thinks objects might be

4. **ROI Head** (Detection):
   - `roi_heads.box_head` - Box regression features
   - `roi_heads.box_predictor` - Final classification features

## Example Workflow

```python
# Complete example: Download, save, and prepare for visualization

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# 1. Get the model
print("Downloading pretrained Faster R-CNN...")
model = fasterrcnn_resnet50_fpn(pretrained=True)

# 2. Save it
print("Saving model...")
torch.save(model.state_dict(), 'faster_rcnn_resnet50_fpn.pth')

# 3. Test it (optional)
model.eval()
from PIL import Image
import torchvision.transforms as T

# Load test image
img = Image.open('test_image.jpg')
transform = T.Compose([T.ToTensor()])
img_tensor = transform(img).unsqueeze(0)

# Run inference
with torch.no_grad():
    predictions = model(img_tensor)

print(f"Detected {len(predictions[0]['boxes'])} objects")
print("Model ready for visualization!")
```

## Visualization Tips

### For Feature Grid Mode:

**Best for**:
- Understanding individual feature detectors
- Seeing specific pattern recognition
- Debugging what the model learns

**Try these layers**:
- Early backbone layers: See edge and texture detection
- FPN layers: See multi-scale feature combination
- RPN conv layers: See where proposals are generated

### For Heatmap Mode:

**Best for**:
- Understanding attention regions
- Seeing where the model focuses
- Identifying important image regions

**Try these layers**:
- Late backbone layers (layer4): Shows high-level semantic focus
- RPN features: Shows potential object locations
- ROI features: Shows regions of interest

## Understanding Faster R-CNN Activations

### Backbone Activations
- **Early layers** (layer1-2): Basic visual features
  - Edges, corners, colors, textures
  - Activated across entire image

- **Middle layers** (layer3): Object parts
  - Wheels, windows, faces, body parts
  - More localized activations

- **Deep layers** (layer4): Object semantics
  - High-level object concepts
  - Very selective activation on specific objects

### FPN Activations
- Combines features from different scales
- **P2** (high resolution): Small objects
- **P3-P4** (medium resolution): Medium objects
- **P5** (low resolution): Large objects

### RPN Activations
- Shows potential object locations
- High activation = "might be an object here"
- Works at multiple scales

## Comparison with Other Models

| Model | Best For | Activation Patterns |
|-------|----------|-------------------|
| **ResNet** | Classification | Global image understanding |
| **YOLO** | Fast detection | Grid-based object presence |
| **Faster R-CNN** | Accurate detection | Region-based object analysis |

## Common Issues and Solutions

### Issue: "Model validation failed"
**Solution**: Make sure you selected the correct backbone architecture that matches your model file.

### Issue: "No extractable layers found"
**Solution**: Faster R-CNN has many nested modules. Try looking at:
- `backbone.body.*` layers
- `backbone.fpn.*` layers
- `rpn.head.*` layers

### Issue: Model is slow
**Solution**:
- Faster R-CNN is larger than ResNet
- Use MobileNet backbone for faster inference
- Consider smaller input images (e.g., 600x600)

### Issue: Activations look unusual
**Solution**:
- Object detection models work differently than classifiers
- RPN layers may show sparse activations (only at object locations)
- This is normal - they're designed to be selective

## Advanced: Analyzing Detection Quality

You can use activation visualization to understand:

1. **Why objects were detected**:
   - Check backbone activations in detected regions
   - See what features triggered the detection

2. **Why objects were missed**:
   - Check if relevant features activated
   - See if RPN proposed regions there

3. **False positives**:
   - Check what features caused incorrect detections
   - Useful for debugging model issues

## Example Use Cases

### Use Case 1: Understanding Car Detection

```
1. Upload Faster R-CNN model
2. Upload street scene image
3. Visualize backbone.body.layer4
   → See high-level "car-ness" features
4. Visualize rpn.head.conv
   → See where model proposes car regions
```

### Use Case 2: Multi-Scale Detection

```
1. Upload image with objects of different sizes
2. Visualize FPN layers (P2-P5)
   → P2: Small objects (pedestrians far away)
   → P3-P4: Medium objects (nearby cars)
   → P5: Large objects (buildings, large vehicles)
```

### Use Case 3: Feature Learning Analysis

```
1. Compare activations across:
   - Early backbone layers (basic features)
   - Middle layers (object parts)
   - Late layers (full objects)
2. Understand the feature hierarchy
```

## Performance Notes

- **Inference time**: 3-8 seconds on CPU (varies with image size)
- **Memory**: Faster R-CNN is larger than ResNet
  - ResNet50 FPN: ~160MB
  - MobileNet V3: ~40MB
- **Works on CPU**: No GPU required (but GPU speeds it up 5-10x)

## Quick Start Checklist

- [ ] Download pretrained model (see Option 1 above)
- [ ] Open the visualization app
- [ ] Select "Faster R-CNN" as model type
- [ ] Choose "resnet50_fpn" as backbone
- [ ] Upload the .pth file
- [ ] Upload a test image with objects
- [ ] Start with `backbone.body.layer4` layer
- [ ] Try both visualization modes
- [ ] Explore RPN and FPN layers

## Resources

**Papers**:
- Faster R-CNN: https://arxiv.org/abs/1506.01497
- Feature Pyramid Networks: https://arxiv.org/abs/1612.03144

**Code**:
- TorchVision Detection: https://pytorch.org/vision/stable/models.html#object-detection

**Datasets** (for pretrained models):
- COCO: 80 object categories
- Custom: Your own trained models

## Summary

Faster R-CNN support adds powerful object detection model visualization to your tool:

✅ Three backbone options (ResNet50, ResNet50-FPN, MobileNet V3)
✅ Visualize backbone, FPN, RPN, and ROI head features
✅ Understand multi-scale object detection
✅ Works on CPU (no GPU needed)
✅ Same easy interface as ResNet and YOLO

Start exploring object detection activations today! 🚀
