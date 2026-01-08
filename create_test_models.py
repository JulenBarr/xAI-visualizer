"""
Script to create properly formatted test models for the XAI Visualizer
This ensures models are saved in the correct format
"""

import torch
from torchvision import models
import os

print("=" * 60)
print("Creating Test Models for XAI Visualizer")
print("=" * 60)
print()

# Create models directory if it doesn't exist
os.makedirs("test_models", exist_ok=True)

# 1. ResNet18 (Small, fast for testing)
print("[1/4] Creating ResNet18...")
try:
    model = models.resnet18(pretrained=True)
    torch.save(model.state_dict(), 'test_models/resnet18_pretrained.pth')
    print("  ✓ Saved: test_models/resnet18_pretrained.pth (~45 MB)")
except Exception as e:
    print(f"  ✗ Error: {e}")

# 2. ResNet50 (Standard, good for most cases)
print("\n[2/4] Creating ResNet50...")
try:
    model = models.resnet50(pretrained=True)
    torch.save(model.state_dict(), 'test_models/resnet50_pretrained.pth')
    print("  ✓ Saved: test_models/resnet50_pretrained.pth (~98 MB)")
except Exception as e:
    print(f"  ✗ Error: {e}")

# 3. Faster R-CNN (Object detection)
print("\n[3/4] Creating Faster R-CNN...")
try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    torch.save(model.state_dict(), 'test_models/faster_rcnn_resnet50_fpn.pth')
    print("  ✓ Saved: test_models/faster_rcnn_resnet50_fpn.pth (~160 MB)")
except Exception as e:
    print(f"  ✗ Error: {e}")

# 4. YOLO (Requires ultralytics)
print("\n[4/4] Creating YOLOv8n...")
try:
    from ultralytics import YOLO
    # This will auto-download if not present
    model = YOLO('yolov8n.pt')
    # Copy to test_models directory
    import shutil
    shutil.copy('yolov8n.pt', 'test_models/yolov8n.pt')
    print("  ✓ Saved: test_models/yolov8n.pt (~6 MB)")
except ImportError:
    print("  ⚠ Skipped: ultralytics not installed")
    print("    Install with: pip install ultralytics")
except Exception as e:
    print(f"  ✗ Error: {e}")

print()
print("=" * 60)
print("Summary")
print("=" * 60)
print()
print("Models are saved in: test_models/")
print()
print("You can now upload these models to the XAI Visualizer:")
print("  1. ResNet18:    Select 'ResNet' + 'resnet18'")
print("  2. ResNet50:    Select 'ResNet' + 'resnet50'")
print("  3. Faster R-CNN: Select 'Faster R-CNN' + 'resnet50_fpn'")
print("  4. YOLOv8n:     Select 'YOLO'")
print()
print("All models saved in CORRECT format (state_dict only)")
print("=" * 60)
