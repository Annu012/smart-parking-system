"""Train YOLOv11 for Parking Detection"""

import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import json

class ParkingTrainer:
    def __init__(self, model_size: str = "m"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
    
    def prepare_dataset_config(self):
        config = {
            "path": str(Path("data").absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 2,
            "names": {0: "empty", 1: "occupied"}
        }
        
        Path("config").mkdir(exist_ok=True)
        with open("config/parking_dataset.yaml", 'w') as f:
            yaml.dump(config, f)
        
        return "config/parking_dataset.yaml"
    
    def train(self, epochs: int = 100, batch_size: int = 16):
        config = self.prepare_dataset_config()
        model = YOLO(f"yolo11{self.model_size}.pt")
        
        print(f"Training YOLOv11-{self.model_size} for Parking Detection")
        results = model.train(
            data=config,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            device=self.device,
            patience=20,
            save=True,
            plots=True
        )
        
        return results
    
    def evaluate(self, model_path: str):
        model = YOLO(model_path)
        metrics = model.val()
        
        return {
            "mAP50": float(metrics.box.map50),
            "mAP": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr)
        }

if __name__ == "__main__":
    trainer = ParkingTrainer(model_size="m")
    results = trainer.train(epochs=100, batch_size=16)
    metrics = trainer.evaluate("runs/detect/train/weights/best.pt")
    
    print("\n=== RESULTS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    Path("results").mkdir(exist_ok=True)
    with open("results/parking_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)