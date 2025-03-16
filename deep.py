import sys
import tempfile
import shutil
import os
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, 
                             QFileDialog, QLabel, QMessageBox, QDialog, QHBoxLayout, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
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
# Book-wise Result Dialog with Table
class BookResultDialog(QDialog):
    def __init__(self, book_results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Book-wise Results")
        self.setGeometry(300, 300, 800, 600)

        layout = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Book Name", "Total Pages", "Detected Pages", "Missing Pages"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setStyleSheet("font-size: 14px; selection-background-color: #85C1E9;")
        layout.addWidget(self.table)

        self.setLayout(layout)
        self.update_results(book_results)

    def update_results(self, book_results):
        self.table.setRowCount(len(book_results))
        for row, (book_name, details) in enumerate(book_results.items()):
            self.table.setItem(row, 0, QTableWidgetItem(book_name))
            self.table.setItem(row, 1, QTableWidgetItem(str(details['total_pages'])))
            self.table.setItem(row, 2, QTableWidgetItem(', '.join(map(str, details['detected_pages']))))
            self.table.setItem(row, 3, QTableWidgetItem(', '.join(map(str, details['missing_pages']))))

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

        # Book-wise Result Button
        self.book_result_btn = QPushButton("Show Book-wise Results")
        self.book_result_btn.setStyleSheet("font-size: 16px; padding: 8px; background-color: #2ecc71; color: white; border-radius: 5px;")
        self.book_result_btn.clicked.connect(self.show_book_wise_results)
        self.book_result_btn.setEnabled(False)
        layout.addWidget(self.book_result_btn)

        # Status Bar
        self.status_bar = self.statusBar()
        self.processing_thread = None
        self.image_paths = {}
        self.results = {}
        self.detected_pages = []

        # Timer for moving progress bar
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_progress_bar)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select PDF Folder")
        if folder:
            self.process_folder(folder)

    def process_folder(self, folder_path):
        self.table.setRowCount(0)
        self.progress_bar.setValue(0)  # Start at 0%
        self.progress_label.setText("Starting...")
        self.book_result_btn.setEnabled(False)

        self.processing_thread = ProcessingThread(folder_path, "best.pt", self.temp_dir)
        self.processing_thread.update_progress.connect(self.update_progress)
        self.processing_thread.result_ready.connect(self.show_results)
        self.processing_thread.error_occurred.connect(self.show_error)
        self.processing_thread.start()

        # Start the timer for moving progress bar
        self.timer.start(100)  # Update every 100ms

    def update_progress(self, value, text):
        self.progress_bar.setValue(value)
        self.progress_label.setText(text)

    def animate_progress_bar(self):
        if self.progress_bar.value() < 100:
            self.progress_bar.setValue(self.progress_bar.value() + 1)
        else:
            self.timer.stop()

    def show_results(self, results, detected_pages):
        self.table.setRowCount(len(results))
        self.results = results
        self.detected_pages = detected_pages

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
        self.book_result_btn.setEnabled(True)

        if detected_pages:
            detected_pages.sort()
            expected_pages = list(range(1, max(detected_pages) + 1))
            missing_pages = sorted(set(expected_pages) - set(detected_pages))
            
            if missing_pages:
                self.status_bar.showMessage(f"Missing pages: {', '.join(map(str, missing_pages))}")
            else:
                self.status_bar.showMessage("All pages accounted for!")

    def show_image(self, index):
        row = index.row()
        image_path = self.image_paths.get(row)
        if image_path:
            viewer = ImageViewer(image_path)
            viewer.exec_()

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def show_book_wise_results(self):
        book_results = self.calculate_book_wise_results()
        dialog = BookResultDialog(book_results, self)
        dialog.exec_()

    def calculate_book_wise_results(self):
        book_results = {}
        for image_path, page_number in self.results.items():
            filename = os.path.basename(image_path)
            book_name = filename.split('.pdf')[0]

            if book_name not in book_results:
                book_results[book_name] = {
                    'total_pages': 0,
                    'detected_pages': [],
                    'missing_pages': [],
                    'lowest_dpi_page': None,
                    'sequential_missing_pages': []
                }

            if page_number.isdigit():
                book_results[book_name]['total_pages'] += 1
                book_results[book_name]['detected_pages'].append(int(page_number))
            else:
                # Handle non-numeric page numbers (e.g., "No page number detected")
                if page_number != "No page number detected":
                    try:
                        # Attempt to extract numeric page number from the string
                        extracted_number = re.search(r'\d+', page_number)
                        if extracted_number:
                            book_results[book_name]['missing_pages'].append(int(extracted_number.group()))
                    except ValueError:
                        # Skip if the page number cannot be converted to an integer
                        pass
        details['missing_pages']=self.newvariable
        for book, details in book_results.items():
            # Ensure missing_pages is not empty before finding sequential missing pages
            if details['missing_pages']:
                details['sequential_missing_pages'] = self.find_sequential_missing_pages(details['missing_pages'])
            else:
                details['sequential_missing_pages'] = []

        return book_results

    def find_sequential_missing_pages(self, missing_pages):
        if not missing_pages:  # Check if the list is empty
            return []  # Return an empty list if there are no missing pages

        missing_pages.sort()
        sequential_missing = []
        start = missing_pages[0]
        prev = start

        for page in missing_pages[1:]:
            if page == prev + 1:
                prev = page
            else:
                if start == prev:
                    sequential_missing.append(start)
                else:
                    sequential_missing.append(f"{start}-{prev}")
                start = page
                prev = page

        if start == prev:
            sequential_missing.append(start)
        else:
            sequential_missing.append(f"{start}-{prev}")

        return sequential_missing

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())