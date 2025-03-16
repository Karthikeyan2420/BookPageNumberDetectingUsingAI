import cv2
from ultralytics import YOLO
import easyocr
import re
import os

class PageNumberDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.ocr = easyocr.Reader(['en'])
        self.page_number_pattern = r'\b\d+\b'

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def detect_page_number(self, image_path, cropped_folder):
        try:
            img = self.preprocess_image(image_path)
            results = self.model.predict(img, conf=0.5)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()

                if len(boxes) == 0:
                    return "No page number detected"

                largest_box = max(boxes, key=lambda box: (box[2]-box[0]) * (box[3]-box[1]))
                x1, y1, x2, y2 = map(int, largest_box)

                cropped = img[y1:y2, x1:x2]
                cropped_image_name = os.path.basename(image_path)
                cropped_image_path = os.path.join(cropped_folder, cropped_image_name)
                cv2.imwrite(cropped_image_path, cropped)

                ocr_result = self.ocr.readtext(cropped)

                for text_entry in ocr_result:
                    text = text_entry[1]
                    if re.search(self.page_number_pattern, text):
                        return text

            return "Page number found but not recognized"

        except Exception as e:
            return f"Error: {str(e)}"

    def process_folder(self, folder_path, cropped_folder):
        os.makedirs(cropped_folder, exist_ok=True)
        results = {}
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(folder_path, filename)
                page_number = self.detect_page_number(image_path, cropped_folder)
                results[filename] = page_number
        return results

# Usage
detector = PageNumberDetector("best.pt")  # Use your trained model
folder_path = "datasets\\data1\\images"
cropped_folder = "cropped"
result = detector.process_folder(folder_path, cropped_folder)

for filename, page_number in result.items():
    print(f"{filename}: Detected Page Number - {page_number}")


# Count total images and successful detections
total_images = len(result)
successful_detections = sum(1 for page_number in result.values() if page_number.isdigit())  # Checks if a valid number was detected

# Calculate percentage
if total_images > 0:
    detection_percentage = (successful_detections / total_images) * 100
else:
    detection_percentage = 0

print(f"Total Images: {total_images}")
print(f"Successfully Detected Page Numbers: {successful_detections}")
print(f"Detection Accuracy: {detection_percentage:.2f}%")