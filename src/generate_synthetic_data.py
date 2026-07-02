import cv2
import numpy as np
from pathlib import Path

# Create dummy parking images (100 images)
output_dir = Path("data/raw/synthetic")
output_dir.mkdir(exist_ok=True)

for i in range(100):
    # Create parking lot image (640x480)
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Draw 10 parking spaces
    for j in range(2):
        for k in range(5):
            x, y = 80 + k*120, 100 + j*200
            if np.random.random() < 0.6:  # 60% occupied
                cv2.rectangle(img, (x, y), (x+100, y+150), (50, 50, 255), -1)
            else:
                cv2.rectangle(img, (x, y), (x+100, y+150), (100, 255, 100), 2)
    
    cv2.imwrite(str(output_dir / f"parking_{i:04d}.jpg"), img)

print(f"✅ Created {len(list(output_dir.glob('*.jpg')))} synthetic parking images")