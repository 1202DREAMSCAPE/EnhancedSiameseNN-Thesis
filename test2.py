import torch

model_path = "Real-ESRGAN/weights/RealESRGAN_x4plus.pth"
try:
    loadnet = torch.load(model_path, map_location=torch.device('cpu'))
    print("✅ Model loaded successfully!")
    print("Model keys:", loadnet.keys())  # Check what keys exist in the file
except Exception as e:
    print("❌ Error loading model:", e)
