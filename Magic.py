# magic_camera_fixed.py - OPTIMIZED FOR CAMERA QUALITY
import cv2
import easyocr
import requests
import numpy as np
import time
import os
from ultralytics import YOLO

print("=== MTG SCANNER - CAMERA OPTIMIZED ===")

# Load model
model = YOLO("mtg_card_detector_enhanced.pt")
print("✅ Model loaded")

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)
print("✅ EasyOCR ready")

def enhance_camera_frame(frame):
    """Significantly enhance camera frame for better OCR"""
    # Convert to LAB color space for better lighting adjustment
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge back and convert to BGR
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Sharpening filter
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    
    return denoised

def detect_cards_robust(frame):
    """Robust detection"""
    # Enhance frame first
    enhanced_frame = enhance_camera_frame(frame)
    
    results = model(enhanced_frame, conf=0.3, iou=0.5, verbose=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence > 0.3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    'coords': (x1, y1, x2, y2),
                    'confidence': confidence
                })
    
    return detections, enhanced_frame

def extract_name_from_camera(cropped):
    """OCR optimized for camera images"""
    if cropped is None or cropped.size == 0:
        return ""
    
    # Save original for debugging
    cv2.imwrite("camera_original.jpg", cropped)
    
    # Apply heavy enhancement for camera images
    enhanced = enhance_camera_frame(cropped)
    cv2.imwrite("camera_enhanced.jpg", enhanced)
    
    h, w = enhanced.shape[:2]
    
    # Try multiple name regions with different sizes
    regions = [
        (0.07, 0.18),   # Top tight region
        (0.05, 0.20),   # Slightly larger
        (0.10, 0.25),   # Lower region (for some layouts)
    ]
    
    all_texts = []
    
    for y_start, y_end in regions:
        start_y = int(h * y_start)
        end_y = int(h * y_end)
        
        if end_y <= start_y:
            continue
            
        name_region = enhanced[start_y:end_y, :]
        
        # Save region for debugging
        cv2.imwrite(f"camera_region_{y_start}.jpg", name_region)
        
        # Resize significantly larger for better OCR
        target_width = 1200  # Much larger for camera quality
        scale = target_width / name_region.shape[1]
        target_height = int(name_region.shape[0] * scale)
        resized = cv2.resize(name_region, (target_width, target_height))
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Heavy preprocessing for camera images
        processed_versions = []
        
        # Method 1: High contrast Otsu
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_versions.append(('otsu', otsu))
        
        # Method 2: Aggressive adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 15, 5)  # More aggressive
        processed_versions.append(('adaptive', adaptive))
        
        # Method 3: Morphological operations to clean text
        kernel = np.ones((2,2), np.uint8)
        morphed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        processed_versions.append(('morphed', morphed))
        
        # Method 4: Inverted threshold (for light text on dark)
        _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processed_versions.append(('otsu_inv', otsu_inv))
        
        for method_name, processed in processed_versions:
            cv2.imwrite(f"camera_{method_name}.jpg", processed)
            
            try:
                # More permissive OCR parameters for camera
                results = reader.readtext(
                    processed,
                    detail=0,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '’-&.,!",
                    text_threshold=0.3,  # Much lower threshold
                    link_threshold=0.2,
                    width_ths=1.0,      # Very permissive spacing
                    batch_size=1,
                    paragraph=False,
                    min_size=5          # Minimum text size
                )
                
                for text in results:
                    clean_text = text.strip()
                    # Very lenient filtering for camera OCR
                    if (2 <= len(clean_text) <= 40 and 
                        sum(c.isalpha() for c in clean_text) >= 2):  # At least 2 letters
                        print(f"   Found: '{clean_text}' (method: {method_name})")
                        all_texts.append(clean_text)
                        
            except Exception as e:
                print(f"   OCR error: {e}")
    
    if not all_texts:
        print("   No text found in any region/method")
        return ""
    
    # Simple selection: prefer longer texts that look like card names
    card_like = []
    for text in all_texts:
        words = text.split()
        # Card names usually have proper capitalization and reasonable length
        if (3 <= len(text) <= 30 and 
            any(w[0].isupper() for w in words if w)):
            card_like.append(text)
    
    if card_like:
        # Return the most card-like text
        return max(card_like, key=len)
    else:
        # Fallback to longest text
        return max(all_texts, key=len)

def get_stable_crop(frame, detection, crop_history=None):
    """Get stable crop by averaging recent detections"""
    x1, y1, x2, y2 = detection['coords']
    
    # Add extra padding specifically for the name area
    card_height = y2 - y1
    extra_top_padding = int(card_height * 0.20)  # More space for name
    extra_side_padding = int(card_height * 0.08)
    extra_bottom_padding = int(card_height * 0.05)
    
    h, w = frame.shape[:2]
    x1 = max(0, x1 - extra_side_padding)
    y1 = max(0, y1 - extra_top_padding)
    x2 = min(w, x2 + extra_side_padding)
    y2 = min(h, y2 + extra_bottom_padding)
    
    cropped = frame[y1:y2, x1:x2]
    
    if cropped.size == 0:
        return None
    
    # Resize to standard size
    return cv2.resize(cropped, (500, 700))  # Larger for better OCR

def fetch_card_details_fuzzy(card_name):
    """Try multiple variations of the card name"""
    if not card_name:
        return None
    
    # Clean and try different variations
    variations = [
        card_name.strip(),
        card_name.strip().title(),  # Title case
        ' '.join(word.capitalize() for word in card_name.strip().split()),  # Each word capitalized
    ]
    
    # Remove duplicates
    variations = list(dict.fromkeys(variations))
    
    for attempt_name in variations:
        print(f"   Trying: '{attempt_name}'")
        
        clean_name = requests.utils.quote(attempt_name)
        url = f"https://api.scryfall.com/cards/named?fuzzy={clean_name}"
        
        try:
            response = requests.get(url, timeout=4)
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Found: {data.get('name', 'Unknown')}")
                return {
                    "name": data.get("name", "Unknown"),
                    "set_name": data.get("set_name", "Unknown"),
                    "mana_cost": data.get("mana_cost", ""),
                }
        except:
            continue
    
    return None

# Main loop with camera optimization
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Higher resolution for better OCR
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
    cap.set(cv2.CAP_PROP_FOCUS, 0)  # Reset focus to auto
    
    last_detected_name = ""
    last_detection_time = 0
    detection_cooldown = 5.0
    stable_detection_count = 0
    
    print("\n🎴 CAMERA-OPTIMIZED SCANNER READY!")
    print("• Ensure GOOD LIGHTING on the card")
    print("• Hold card STEADY and FLAT")
    print("• Fill frame with the card")
    print("• Make sure camera is IN FOCUS")
    print("• Press 'q' to quit")
    print("• Check debug images if issues persist\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Detect cards on enhanced frame
        detections, enhanced_frame = detect_cards_robust(frame)
        current_time = time.time()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['coords']
            confidence = detection['confidence']
            
            # Draw on original frame for display
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"Card {confidence:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Only process if we have high confidence and stable detection
            if (confidence > 0.8 and 
                current_time - last_detection_time > detection_cooldown):
                
                stable_detection_count += 1
                
                # Wait for stable detection (card held still)
                if stable_detection_count >= 2:
                    print(f"\n🔍 Processing stable detection (conf: {confidence:.2f})...")
                    
                    # Get enhanced crop
                    cropped_card = get_stable_crop(enhanced_frame, detection)
                    
                    if cropped_card is not None:
                        print("   Running enhanced OCR...")
                        card_name = extract_name_from_camera(cropped_card)
                        
                        if card_name and card_name != last_detected_name:
                            print(f"   📖 OCR result: '{card_name}'")
                            
                            # Try to find card with fuzzy matching
                            details = fetch_card_details_fuzzy(card_name)
                            
                            if details:
                                print(f"   ✅ IDENTIFIED: {details['name']}")
                                print(f"      Set: {details['set_name']}")
                                if details['mana_cost']:
                                    print(f"      Cost: {details['mana_cost']}")
                                
                                # Show success
                                cv2.putText(frame, f"FOUND: {details['name']}", 
                                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.8, (0, 255, 0), 2)
                                last_detected_name = details['name']
                            else:
                                print("   ❌ No match found")
                                cv2.putText(frame, f"OCR: {card_name}", 
                                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.7, (0, 165, 255), 2)
                                last_detected_name = card_name
                            
                            last_detection_time = current_time
                            stable_detection_count = 0
            else:
                stable_detection_count = 0
        
        # Display guidance
        cv2.putText(frame, f"Detections: {len(detections)} | Hold steady to scan", 
                   (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Last: {last_detected_name[:25]}", 
                   (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'q' to quit", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        cv2.imshow("MTG Scanner - CAMERA OPTIMIZED", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Scanner closed")

if __name__ == "__main__":
    main()