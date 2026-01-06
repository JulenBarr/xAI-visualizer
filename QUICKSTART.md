# Quick Start Guide

## Installation

1. Install all dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. The app will open in your browser at `http://localhost:8501`

## Testing with a Sample Model

### Option 1: Using a Pretrained ResNet

Create a simple script to download a pretrained ResNet50:

```python
# download_resnet.py
import torch
from torchvision import models

# Download pretrained ResNet50
model = models.resnet50(pretrained=True)

# Save the model
torch.save(model.state_dict(), 'resnet50_pretrained.pth')
print("Model saved as resnet50_pretrained.pth")
```

Run it:
```bash
python download_resnet.py
```

Then in the app:
1. Select "ResNet" as model type
2. Select "resnet50" as architecture
3. Upload `resnet50_pretrained.pth`
4. Upload any image
5. Visualize!

### Option 2: Using YOLO

Download a YOLO model:

```python
# download_yolo.py
from ultralytics import YOLO

# Download YOLOv8 nano (smallest, fastest)
model = YOLO('yolov8n.pt')
print("Model downloaded as yolov8n.pt")
```

Run it:
```bash
python download_yolo.py
```

Then in the app:
1. Select "YOLO" as model type
2. Upload `yolov8n.pt`
3. Upload any image
4. Visualize!

## Sample Images

You can use any image from the internet or your own photos. Good test images:
- Photos of animals (dogs, cats, birds)
- Street scenes
- Objects (cars, furniture, etc.)

## Recommended Layers to Visualize

### For ResNet:
- Early layers: `layer1.0.conv1` - detects edges and simple patterns
- Middle layers: `layer2.0.conv2` - detects more complex features
- Deep layers: `layer4.2.conv3` - detects high-level semantic features

### For YOLO:
- Explore the `model.0` through `model.9` layers for different detection scales

## Tips

1. **Start with Feature Grid** to see individual channels
2. **Use Aggregated Heatmap** to see what the model focuses on
3. **Try different layers** to see how features become more abstract
4. **Compare early vs late layers** to understand the feature hierarchy

## Troubleshooting

- If the app doesn't start, make sure all dependencies are installed
- If model loading fails, check that the file format matches the model type
- For memory issues, try smaller images or visualize layers with fewer channels

## Next Steps

Once you're comfortable with the basics:
- Try your own trained models
- Compare activations across different images
- Explore how different architectures "see" the same image differently
