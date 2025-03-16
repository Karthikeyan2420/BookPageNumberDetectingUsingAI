import cv2
from ultralytics import YOLO
import easyocr
import re
class PageNumberDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.ocr = easyocr.Reader(['en'])
        self.page_number_pattern = r'\b\d+\b'

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def detect_page_number(self, image_path):
        try:
            # Detect page number region
            img = self.preprocess_image(image_path)
            results = self.model.predict(img, conf=0.5)
            
            # Process detections
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                
                if len(boxes) == 0:
                    return "No page number detected"

                # Get largest detection (assuming page number is prominent)
                largest_box = max(boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
                x1, y1, x2, y2 = map(int, largest_box)
                
                # Crop and OCR
                cropped = img[y1:y2, x1:x2]
                ocr_result = self.ocr.readtext(cropped)
                cv2.imwrite("h.png",cropped)
                
                # Filter numerical results
                for text_entry in ocr_result:
                    text = text_entry[1]
                    if re.search(self.page_number_pattern, text):
                        return text
                
            return "Page number found but not recognized"

        except Exception as e:
            return f"Error: {str(e)}"

# Usage
detector = PageNumberDetector("best.pt")  # Use your trained model
result = detector.detect_page_number("datasets\\data\\images\\val\\book8_page_261.png")
print("Detected Page Number:", result)