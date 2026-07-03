"""Real-time Parking Detection"""

import cv2, numpy as np, time
from ultralytics import YOLO

class RealtimeParkingInference:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def process_frame(self, frame: np.ndarray) -> dict:
        start = time.time()
        results = self.model(frame, conf=0.5, verbose=False)
        inference_time = time.time() - start

        empty = occupied = 0
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0: occupied += 1   # 0 = occupied
                else: empty += 1              # 1 = empty

        return {
            "empty": empty,
            "occupied": occupied,
            "inference_time_ms": inference_time * 1000,
            "fps": 1 / (inference_time + 1e-6)
        }


if __name__ == "__main__":
    inf = RealtimeParkingInference("runs/detect/train-5/weights/best.pt")

    # --- Validation metrics (mAP50, Precision, Recall) ---
    metrics = inf.model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print()

    # --- Real-time inference loop ---
    cap = cv2.VideoCapture(0)  # webcam; swap for a video/image path if needed
    if not cap.isOpened():
        raise RuntimeError("Could not open camera/video source")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = inf.process_frame(frame)
        print(f"✅ Empty: {result['empty']}, Occupied: {result['occupied']}, "
              f"Inference: {result['inference_time_ms']:.1f}ms, FPS: {result['fps']:.1f}")

        cv2.imshow("Parking Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()