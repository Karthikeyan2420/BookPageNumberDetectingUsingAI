import sys
import os
import tempfile
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from pdf2image import convert_from_path
import cv2
from ultralytics import YOLO
import easyocr
import re
from PIL import Image
import PyPDF2

# Detect GPU availability
USE_GPU = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 4000000000  # 4GB+

class PageNumberDetector:
    def __init__(self, model_path):
        self.device = 'cuda' if USE_GPU else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        self.ocr = easyocr.Reader(['en'], gpu=USE_GPU)
        self.page_number_pattern = r'\b\d+\b'

    def detect_page_number(self, image_path):
        try:
            img = cv2.imread(image_path)
            results = self.model.predict(img, conf=0.5, device=self.device)
            
            if not results or len(results[0].boxes) == 0:
                return None, "No page number detected"
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            x1, y1, x2, y2 = map(int, max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))
            cropped = img[y1:y2, x1:x2]
            ocr_result = self.ocr.readtext(cropped)
            
            for text_entry in ocr_result:
                text = text_entry[1]
                if re.search(self.page_number_pattern, text):
                    return text, None
            
            return None, "Page number found but not recognized"
        except Exception as e:
            return None, f"Error: {str(e)}"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Page Number Detector")
        self.setGeometry(100, 100, 1200, 800)
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = "best.pt"
        self.detector = PageNumberDetector(self.model_path)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.browse_btn = QPushButton("📂 Select PDF Folder")
        self.browse_btn.setStyleSheet("font-size: 16px; padding: 8px;")
        self.browse_btn.clicked.connect(self.browse_folder)

        self.progress_label = QLabel("Waiting for process...")
        self.progress_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Preview", "Filename", "Page Number", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        
        layout.addWidget(self.browse_btn)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.table)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select PDF Folder")
        if folder:
            self.process_folder(folder)

    def process_folder(self, folder_path):
        self.table.setRowCount(0)
        self.progress_label.setText("Processing...")
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            self.convert_and_detect(pdf_path)
        
        self.progress_label.setText("Processing Complete")

    def convert_and_detect(self, pdf_path):
        try:
            images = convert_from_path(pdf_path, dpi=300)
            for img_idx, img in enumerate(images):
                image_path = os.path.join(self.temp_dir, f"{os.path.basename(pdf_path)}_{img_idx}.jpg")
                img.save(image_path, "JPEG", dpi=(300, 300))
                page_number, error = self.detector.detect_page_number(image_path)
                
                self.update_table(image_path, os.path.basename(pdf_path), img_idx + 1, page_number, error)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def update_table(self, image_path, filename, page_index, detected_page, error):
        row = self.table.rowCount()
        self.table.insertRow(row)

        preview = QLabel()
        pixmap = QPixmap(image_path).scaled(100, 100, Qt.KeepAspectRatio)
        preview.setPixmap(pixmap)
        self.table.setCellWidget(row, 0, preview)

        self.table.setItem(row, 1, QTableWidgetItem(filename))
        self.table.setItem(row, 2, QTableWidgetItem(str(detected_page if detected_page else "N/A")))
        
        status = "✅" if str(page_index) == str(detected_page) else "❌"
        self.table.setItem(row, 3, QTableWidgetItem(status))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
