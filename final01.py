import sys
import tempfile
import shutil
import os
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, 
                             QFileDialog, QLabel, QMessageBox, QDialog, QHBoxLayout, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont
from pdf2image import convert_from_path
import cv2
from ultralytics import YOLO
import easyocr
import re

# Detect if GPU is available
USE_GPU = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 4000000000  # 4GB+

# Page Number Detector (Sequential Processing)
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
                return "No page number detected"

            boxes = results[0].boxes.xyxy.cpu().numpy()
            x1, y1, x2, y2 = map(int, max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1])))

            cropped = img[y1:y2, x1:x2]
            ocr_result = self.ocr.readtext(cropped)

            for text_entry in ocr_result:
                text = text_entry[1]
                if re.search(self.page_number_pattern, text):
                    return text

            return "Page number found but not recognized"
        except Exception as e:
            return f"Error: {str(e)}"

# Processing Thread
class ProcessingThread(QThread):
    update_progress = pyqtSignal(int, str)
    result_ready = pyqtSignal(dict, list)
    error_occurred = pyqtSignal(str)

    def __init__(self, folder_path, model_path, temp_dir):
        super().__init__()
        self.folder_path = folder_path
        self.model_path = model_path
        self.detector = PageNumberDetector(model_path)
        self.temp_dir = temp_dir

    def run(self):
        try:
            pdf_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.pdf')]
            results = {}
            detected_pages = []
            total_steps = len(pdf_files) * 2
            step_count = 0

            for pdf_file in pdf_files:
                self.update_progress.emit(int((step_count / total_steps) * 100), "Converting PDF to Images...")
                pdf_path = os.path.join(self.folder_path, pdf_file)
                images = convert_from_path(pdf_path, dpi=300)

                for img_idx, img in enumerate(images):
                    full_image_path = os.path.join(self.temp_dir, f"{pdf_file}_{img_idx}.jpg")
                    img.save(full_image_path, "JPEG")
                    results[full_image_path] = None
                step_count += 1

            for image_path in results.keys():
                self.update_progress.emit(int((step_count / total_steps) * 100), "Detecting Page Numbers...")
                page_number = self.detector.detect_page_number(image_path)
                results[image_path] = page_number
                if page_number.isdigit():
                    detected_pages.append(int(page_number))
                step_count += 1

            self.result_ready.emit(results, detected_pages)

        except Exception as e:
            self.error_occurred.emit(str(e))

# Image Viewer
class ImageViewer(QDialog):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Full Page Viewer")
        self.setGeometry(200, 200, 800, 1000)

        layout = QVBoxLayout()
        self.label = QLabel()
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.label.setText("Error: Image could not be loaded!")
        else:
            self.label.setPixmap(pixmap.scaled(800, 1000, Qt.KeepAspectRatio))
        layout.addWidget(self.label)
        self.setLayout(layout)

# Main GUI Application
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Page Number Detector")
        self.setGeometry(100, 100, 1200, 800)
        
        self.temp_dir = tempfile.mkdtemp()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Browse & Process Buttons
        self.browse_btn = QPushButton("📂 Select PDF Folder")
        self.browse_btn.setStyleSheet("font-size: 16px; padding: 8px; background-color: #3498db; color: white; border-radius: 5px;")
        self.browse_btn.clicked.connect(self.browse_folder)
        
        layout.addWidget(self.browse_btn)

        # Progress Bar & Status
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Waiting for process...")
        self.progress_label.setFont(QFont("Arial", 12, QFont.Bold))
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)

        layout.addLayout(progress_layout)

        # Results Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Preview", "Filename", "Page Number", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setStyleSheet("font-size: 14px; selection-background-color: #85C1E9;")
        layout.addWidget(self.table)

        # Status Bar
        self.status_bar = self.statusBar()
        self.processing_thread = None
        self.image_paths = {}

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select PDF Folder")
        if folder:
            self.process_folder(folder)

    def process_folder(self, folder_path):
        self.table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting...")

        self.processing_thread = ProcessingThread(folder_path, "best.pt", self.temp_dir)
        self.processing_thread.update_progress.connect(self.update_progress)
        self.processing_thread.result_ready.connect(self.show_results)
        self.processing_thread.error_occurred.connect(self.show_error)
        self.processing_thread.start()

    def update_progress(self, value, text):
        self.progress_bar.setValue(value)
        self.progress_label.setText(text)

    def show_results(self, results, detected_pages):
        self.table.setRowCount(len(results))

        for row, (image_path, page_number) in enumerate(results.items()):
            filename = os.path.basename(image_path)
            self.image_paths[row] = image_path

            preview_label = QLabel()
            pixmap = QPixmap(image_path)
            preview_label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))
            self.table.setCellWidget(row, 0, preview_label)

            self.table.setItem(row, 1, QTableWidgetItem(filename))
            self.table.setItem(row, 2, QTableWidgetItem(str(page_number)))

            status_label = QLabel("✅" if page_number.isdigit() else "❌")
            self.table.setCellWidget(row, 3, status_label)

        self.progress_label.setText("Processing Complete")
        self.table.doubleClicked.connect(self.show_image)

    def show_image(self, index):
        row = index.row()
        image_path = self.image_paths.get(row)
        if image_path:
            viewer = ImageViewer(image_path)
            viewer.exec_()

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
