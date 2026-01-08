"""
Quick script to test YOLO model loading and debug the issue
"""

import torch
import torch.nn as nn

try:
    from ultralytics import YOLO
    print("✓ Ultralytics imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ultralytics: {e}")
    print("Install with: pip install ultralytics")
    exit(1)

# Test with a small YOLO model
print("\n--- Testing YOLO Model Loading ---")

try:
    # Download YOLOv8 nano (smallest model)
    print("Downloading YOLOv8n (this may take a moment)...")
    yolo = YOLO('yolov8n.pt')
    print(f"✓ YOLO object created: {type(yolo)}")

    # Check what attributes it has
    print(f"\nYOLO object attributes:")
    print(f"  - Has 'model' attribute: {hasattr(yolo, 'model')}")
    print(f"  - Has 'pt' attribute: {hasattr(yolo, 'pt')}")
    print(f"  - Has 'predictor' attribute: {hasattr(yolo, 'predictor')}")

    # Try to get the model
    if hasattr(yolo, 'model'):
        model = yolo.model
        print(f"\n✓ Model extracted: {type(model)}")
        print(f"  - Is nn.Module: {isinstance(model, nn.Module)}")

        if isinstance(model, nn.Module):
            model.eval()
            print("✓ Model set to eval mode")

            # Try to get layers
            print(f"\n✓ Model has {len(list(model.named_modules()))} modules")

            # Show first few module names
            print("\nFirst 10 module names:")
            for i, (name, module) in enumerate(list(model.named_modules())[:10]):
                print(f"  {i}. {name} -> {type(module).__name__}")

            print("\n✅ YOLO model loaded successfully!")
            print(f"Model file saved at: yolov8n.pt")

        else:
            print(f"✗ Model is not nn.Module: {type(model)}")
    else:
        print("✗ YOLO object doesn't have 'model' attribute")
        print(f"Available attributes: {dir(yolo)}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Test Complete ---")
