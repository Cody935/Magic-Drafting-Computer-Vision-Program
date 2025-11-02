# test.py - Debug YOLO
from ultralytics import YOLO
import cv2

model = YOLO("mtg_card_detector.pt")

img = cv2.imread("images/ayara.jpg")  # ← Change to any image
results = model(img, conf=0.1, verbose=True)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)
        conf = box.conf.item()
        print(f"Detected: class={cls}, conf={conf:.2f}, box=({x1},{y1},{x2},{y2})")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()