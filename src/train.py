from ultralytics import YOLO
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
# Load model
model = YOLO('yolo11n.pt')  # nano for quick testing
# Train
results = model.train(
    data='config/parking.yaml',
    epochs=10,  # Just 10 for Day 5
    imgsz=640,
    batch=8,
    patience=5,
    device=0 if torch.cuda.is_available() else 'cpu'
)
print("Training complete!")
print(f"mAP50: {results.results_dict.get('metrics/mAP50', 0):.4f}")
