from ultralytics import YOLO
import os
import shutil

# Load model
model = YOLO("yolov8n.pt")

print("Starting training on 55 MTG cards...")

results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="mtg_draft",
    patience=15
)

# === FIXED: Save best model manually ===
weights_dir = "runs/detect/mtg_draft/weights"
best_pt = os.path.join(weights_dir, "best.pt")

if os.path.exists(best_pt):
    print(f"\nTraining complete! Best model: {best_pt}")
    shutil.copy(best_pt, "mtg_card_detector.pt")
    print("Model copied to: mtg_card_detector.pt")
else:
    last_pt = os.path.join(weights_dir, "last.pt")
    print(f"Using last model: {last_pt}")
    shutil.copy(last_pt, "mtg_card_detector.pt")

print("\nYou can now run: python Magic.py")