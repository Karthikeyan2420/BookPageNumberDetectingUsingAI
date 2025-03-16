""" import sys
import tempfile
import shutil
import os
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QLabel, 
                             QMessageBox, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from pdf2image import convert_from_path
import cv2
from ultralytics import YOLO
import easyocr
import re

USE_GPU = torch.cuda.is_available()

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

class ProcessingThread(QThread):
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, folder_path, model_path, temp_dir):
        super().__init__()
        self.folder_path = folder_path
        self.model_path = model_path
        self.detector = PageNumberDetector(model_path)
        self.temp_dir = temp_dir
        self.running = True

    def run(self):
        try:
            pdf_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.pdf')]
            results = {}

            for pdf_file in pdf_files:
                if not self.running:
                    return
                pdf_path = os.path.join(self.folder_path, pdf_file)
                images = convert_from_path(pdf_path, dpi=300)

                for img_idx, img in enumerate(images):
                    if not self.running:
                        return
                    full_image_path = os.path.join(self.temp_dir, f"{pdf_file}_{img_idx}.jpg")
                    img.save(full_image_path, "JPEG")
                    results[full_image_path] = self.detector.detect_page_number(full_image_path)

            self.result_ready.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self.running = False

class BookResultDialog(QDialog):
    def __init__(self, book_results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Book-wise Results")
        self.setGeometry(300, 300, 800, 600)
        layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Book Name", "Total Pages", "Missing Pages", "Non-Sequential Pages"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.update_results(book_results)

    def update_results(self, book_results):
        self.table.setRowCount(len(book_results))
        for row, (book_name, details) in enumerate(book_results.items()):
            self.table.setItem(row, 0, QTableWidgetItem(book_name))
            self.table.setItem(row, 1, QTableWidgetItem(str(details['total_pages'])))
            self.table.setItem(row, 2, QTableWidgetItem(', '.join(map(str, details['missing_pages']))))
            self.table.setItem(row, 3, QTableWidgetItem(', '.join(details['non_sequential_pages'])))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Page Number Detector")
        self.setGeometry(100, 100, 800, 600)
        self.temp_dir = tempfile.mkdtemp()
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.browse_btn = QPushButton("Select PDF Folder")
        self.browse_btn.clicked.connect(self.browse_folder)
        layout.addWidget(self.browse_btn)

        self.cancel_btn = QPushButton("Cancel Processing")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        layout.addWidget(self.cancel_btn)

        self.book_result_btn = QPushButton("Show Book-wise Results")
        self.book_result_btn.setEnabled(False)
        self.book_result_btn.clicked.connect(self.show_book_wise_results)
        layout.addWidget(self.book_result_btn)

        self.status_label = QLabel("Waiting for process...")
        layout.addWidget(self.status_label)

        self.processing_thread = None
        self.results = {}

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select PDF Folder")
        if folder:
            self.process_folder(folder)

    def process_folder(self, folder_path):
        self.browse_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("Processing...")
        self.processing_thread = ProcessingThread(folder_path, "best.pt", self.temp_dir)
        self.processing_thread.result_ready.connect(self.show_results)
        self.processing_thread.error_occurred.connect(self.show_error)
        self.processing_thread.start()

    def cancel_processing(self):
        if self.processing_thread:
            self.processing_thread.stop()
        self.browse_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Processing Cancelled.")

    def show_results(self, results):
        self.results = results
        self.browse_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Processing Complete.")
        self.book_result_btn.setEnabled(True)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def show_book_wise_results(self):
        dialog = BookResultDialog(self.results, self)
        dialog.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
 """


