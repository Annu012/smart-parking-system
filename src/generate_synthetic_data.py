import cv2, numpy as np
from pathlib import Path

for split in ['train', 'val', 'test']:
    for dir_type in ['images', 'labels']:
        Path(f"data/{split}/{dir_type}").mkdir(parents=True, exist_ok=True)
    
    num = 100 if split == 'train' else 20
    for i in range(num):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 180
        labels = []
        for row in range(4):
            for col in range(5):
                x, y, w, h = 70+col*120, 80+row*120, 100, 100
                occupied = np.random.random() < 0.65
                cv2.rectangle(img, (x, y), (x+w, y+h), (50, 100, 200) if occupied else (0, 200, 100), -1 if occupied else 2)
                cls = 1 if occupied else 0
                labels.append(f"{cls} {(x+w/2)/640:.4f} {(y+h/2)/480:.4f} {w/640:.4f} {h/480:.4f}")
        
        cv2.imwrite(f"data/{split}/images/img_{i:04d}.jpg", img)
        with open(f"data/{split}/labels/img_{i:04d}.txt", 'w') as f:
            f.write('\n'.join(labels))

print("✅ Data ready")