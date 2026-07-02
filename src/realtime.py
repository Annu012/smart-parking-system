import cv2
import numpy as np
from ultralytics import YOLO
import glob
import os

class ParkingDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            candidates = glob.glob('runs/detect/*/weights/best.pt')
            if not candidates:
                raise FileNotFoundError("No trained model found in runs/detect/*/weights/best.pt")
            model_path = max(candidates, key=os.path.getmtime)
            print(f"Using latest model: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = {0: 'occupied', 1: 'empty'}

    def process_image(self, image_path: str) -> dict:
        """Process single image"""
        img = cv2.imread(image_path)
        results = self.model(img, conf=0.5)

        empty = 0
        occupied = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    occupied += 1
                else:
                    empty += 1

        total = empty + occupied
        occupancy = occupied / (total + 1e-6) if total > 0 else 0

        return {
            'empty': empty,
            'occupied': occupied,
            'occupancy_rate': occupancy,
            'total': total
        }

# Test
detector = ParkingDetector()
result = detector.process_image('data/val/images/parking_0000.jpg')
print(f"Empty: {result['empty']}, Occupied: {result['occupied']}")
print(f"Occupancy: {result['occupancy_rate']:.1%}")