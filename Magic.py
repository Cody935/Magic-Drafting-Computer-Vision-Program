# Magic.py - YOLOv8 + EasyOCR + Scryfall
import cv2
import easyocr
import requests
import numpy as np
import time
from ultralytics import YOLO

# -------------------------------
# 1. LOAD TRAINED YOLO MODEL
# -------------------------------
print("Loading YOLOv8 MTG Card Detector...")
model = YOLO("mtg_card_detector.pt")  # ← Your trained model
# If you don't have it yet, we'll generate it below

# -------------------------------
# 2. EASYOCR
# -------------------------------
reader = easyocr.Reader(['en'], gpu=False)
_ = reader.readtext(np.zeros((50, 200, 3), dtype=np.uint8), detail=0)

# -------------------------------
# 3. PERSPECTIVE WARP (from YOLO box)
# -------------------------------
def warp_card(frame, box):
    x1, y1, x2, y2 = map(int, box)
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")
    width = max(x2 - x1, y2 - y1) * 1.3
    height = width * 1.4  # 2:3 ratio

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(frame, M, (int(width), int(height)))
    return warped

# -------------------------------
# 4. OCR (same as before, but on warped card)
# -------------------------------
def extract_card_name(warped):
    h, w = warped.shape[:2]
    if min(h, w) < 100:
        return ""

    name_h = int(h * 0.25)
    name_region = warped[0:name_h, :]

    target_w = max(600, w)
    scale = target_w / w
    name_region = cv2.resize(name_region, (target_w, int(name_h * scale)), interpolation=cv2.INTER_CUBIC)

    # Denoise + sharpen
    name_region = cv2.fastNlMeansDenoisingColored(name_region, None, 10, 10, 7, 21)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    name_region = cv2.filter2D(name_region, -1, kernel)

    lab = cv2.cvtColor(name_region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    name_region = cv2.merge([l, a, b])
    name_region = cv2.cvtColor(name_region, cv2.COLOR_LAB2BGR)
    name_rgb = cv2.cvtColor(name_region, cv2.COLOR_BGR2RGB)

    results = reader.readtext(
        name_rgb, detail=0,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '’-&.,",
        contrast_ths=0.2, text_threshold=0.7
    )

    if not results:
        return ""
    candidates = [s.strip() for s in results if len(s.strip()) > 3]
    if not candidates:
        return results[0].strip()
    return max(candidates, key=lambda x: (len(x), ' ' in x or '-' in x))

# -------------------------------
# 5. SCRYFALL
# -------------------------------
def fetch_card_details(name):
    url = f"https://api.scryfall.com/cards/named?fuzzy={name}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            d = r.json()
            return {k: d.get(k) for k in ["name", "set_name", "rarity", "oracle_text"]}
    except:
        pass
    return None

# -------------------------------
# 6. MAIN LOOP
# -------------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280); cap.set(4, 720)

last_name = ""
last_time = 0

print("MTG Card Scanner READY! Hold any card in view.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25, iou=0.5, verbose=False)
    card_detected = False

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # class 0 = "mtg_card"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "MTG CARD", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                warped = warp_card(frame, [x1, y1, x2, y2])
                name = extract_card_name(warped)

                if name and name != last_name and time.time() - last_time > 1.5:
                    print(f"\nDetected: {name}")
                    details = fetch_card_details(name)
                    if details:
                        for k, v in details.items():
                            print(f"{k}: {v}")
                    else:
                        print("Not found.")
                    last_name = name
                    last_time = time.time()

                card_detected = True

    cv2.imshow("YOLOv8 MTG Scanner", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()