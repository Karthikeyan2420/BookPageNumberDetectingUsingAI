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
from PyQt5.QtWidgets import QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage  

# Detect if GPU is available
USE_GPU = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 4000000000  # 4GB+

# Page Number Detector (Sequential Processing)
class PageNumberDetector:
    def __init__(self, model_path):
        self.device = 'cuda' if USE_GPU else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        self.ocr = easyocr.Reader(['en'], gpu=USE_GPU)
        self.page_number_pattern = r'\b\d+\b|\b[IVXLCDM]+\b'  # Updated to include Roman numerals

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
                self.update_progress.emit(int((step_count / total_steps) * 100), pdf_file+" - Converting PDF to Images...")
                pdf_path = os.path.join(self.folder_path, pdf_file)
                images = convert_from_path(pdf_path, dpi=300,thread_count=15)

                for img_idx, img in enumerate(images):
                    full_image_path = os.path.join(self.temp_dir, f"{pdf_file}_{img_idx}.jpg")
                    img.save(full_image_path, "JPEG")
                    results[full_image_path] = None
                step_count += 1

            for image_path in results.keys():
                self.update_progress.emit(int((step_count / total_steps) * 100), "Detecting Page Numbers...")
                page_number = self.detector.detect_page_number(image_path)
                results[image_path] = page_number
                if re.match(r'\b\d+\b|\b[IVXLCDM]+\b', page_number):  # Check for both Arabic and Roman numerals
                    detected_pages.append(page_number)
                step_count += 1

            self.result_ready.emit(results, detected_pages)

        except Exception as e:
            self.error_occurred.emit(str(e))

from PyQt5.QtWidgets import QSlider
from PyQt5.QtCore import Qt

class ImageViewer(QDialog):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Full Page Viewer")
        self.setGeometry(200, 200, 800, 1000)

        # Main layout
        layout = QVBoxLayout()

        # Image label
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # Zoom slider
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(50)  # 50% zoom
        self.zoom_slider.setMaximum(200)  # 200% zoom
        self.zoom_slider.setValue(100)  # Default zoom (100%)
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        layout.addWidget(self.zoom_slider)

        # Set layout
        self.setLayout(layout)

        # Load the image
        self.image_path = image_path
        self.original_pixmap = QPixmap(image_path)
        if self.original_pixmap.isNull():
            self.label.setText("Error: Image could not be loaded!")
        else:
            # Set medium size for the preview (e.g., 50% of original size)
            self.medium_size = self.original_pixmap.size() * 0.5
            self.label.setPixmap(self.original_pixmap.scaled(self.medium_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_zoom(self):
        """Update the image size based on the zoom slider value."""
        if not self.original_pixmap.isNull():
            zoom_factor = self.zoom_slider.value() / 100.0
            new_size = self.medium_size * zoom_factor
            self.label.setPixmap(self.original_pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

# Book-wise Result Dialog with Table
class BookResultDialog(QDialog):
    def __init__(self, book_results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Book-wise Results")
        self.setGeometry(300, 300, 1000, 600)  # Increased width to accommodate more columns

        layout = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setColumnCount(5)  # Added columns for All Pages Above 300 DPI and In-Order Pages
        self.table.setHorizontalHeaderLabels(["Book Name", "Page Number", "Missing Pages", "All Pages Above 300 DPI", "In-Order Pages"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet("font-size: 14px; selection-background-color: #85C1E9;")
        layout.addWidget(self.table)

        self.setLayout(layout)
        self.update_results(book_results)

    def update_results(self, book_results):
        self.table.setRowCount(len(book_results))
        for row, (book_name, details) in enumerate(book_results.items()):
            self.table.setItem(row, 0, QTableWidgetItem(book_name))
            self.table.setItem(row, 1, QTableWidgetItem(', '.join(map(str, details['detected_pages']))))
            self.table.setItem(row, 2, QTableWidgetItem(', '.join(map(str, details['missing_pages']))))
            self.table.setItem(row, 3, QTableWidgetItem("Yes" if details['all_pages_above_300dpi'] else "No"))
            self.table.setItem(row, 4, QTableWidgetItem("Yes" if details['in_order_pages'] else "No"))


# Main GUI Application
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Page Number Detector")
        self.setGeometry(100, 100, 1200, 800)
        
        self.temp_dir = tempfile.mkdtemp()
        self.is_processing=False
        self.is_finished=False

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Browse & Process Buttons
        self.browse_btn = QPushButton("📂 Select PDF Folder")
        self.browse_btn.setStyleSheet("font-size: 16px; padding: 8px; background-color: #3498db; color: white; border-radius: 5px;")
        self.browse_btn.clicked.connect(self.browse_folder)
        
        # Cancel Button
        self.cancel_btn = QPushButton("❌ Cancel Process")
        self.cancel_btn.setStyleSheet("font-size: 16px; padding: 8px; background-color: #e74c3c; color: white; border-radius: 5px;")
        self.cancel_btn.clicked.connect(self.cancel_process)
        self.cancel_btn.setEnabled(False)  # Disabled by default

        # Add buttons to layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.browse_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

        # Progress Bar & Status
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("Please upload folder to strat the process...")
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

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select PDF Folder")
        if folder:
            self.process_folder(folder)

    def process_folder(self, folder_path):
        self.table.setRowCount(0)
        self.progress_bar.setValue(0)  # Start at 0%
        self.progress_label.setText("Starting...")
        self.book_result_btn.setEnabled(False)
        self.is_processing = True
        self.is_finished = False
        self.browse_btn.setEnabled(False)  # Disable browse button during processing
        self.cancel_btn.setEnabled(True)

        self.processing_thread = ProcessingThread(folder_path, "best.pt", self.temp_dir)
        self.processing_thread.update_progress.connect(self.update_progress)
        self.processing_thread.result_ready.connect(self.show_results)
        self.processing_thread.error_occurred.connect(self.show_error)
        self.processing_thread.start()

    def update_progress(self, value, text):
        self.progress_bar.setValue(value)
        self.progress_label.setText(text)
    def cancel_process(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.terminate()  # Terminate the thread
            self.processing_thread.wait()       # Wait for the thread to finish
            self.reset_state()
            self.status_bar.showMessage("Process canceled.")
    def reset_state(self):
        self.is_processing = False
        self.is_finished = True
        self.browse_btn.setEnabled(True)  # Re-enable browse button
        self.cancel_btn.setEnabled(False)  # Disable cancel button
        self.progress_bar.setValue(0)
        self.progress_label.setText("Please upload folder to strat the process...")

    def show_results(self, results, detected_pages):
        self.is_processing = False
        self.is_finished = True
        self.browse_btn.setEnabled(True)  # Re-enable browse button
        self.cancel_btn.setEnabled(False)
        self.table.setRowCount(len(results))
        self.results = results
        self.detected_pages = detected_pages

        for row, (image_path, page_number) in enumerate(results.items()):
            filename = os.path.basename(image_path)
            self.image_paths[row] = image_path

            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Unable to load image at {image_path}")
                continue

            # Get the bounding box for the detected page number (if any)
            results = self.processing_thread.detector.model.predict(img, conf=0.5)
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                x1, y1, x2, y2 = map(int, max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1])))
                cropped_img = img[y1:y2, x1:x2]
            else:
                cropped_img = img  # Fallback to full image if no bounding box is found

            # Convert the cropped image to QPixmap for display
            cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            height, width, channel = cropped_img_rgb.shape
            bytes_per_line = channel * width
            q_img = QPixmap.fromImage(QImage(cropped_img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888))

            # Create a QLabel for the preview
            preview_label = QLabel()
            preview_label.setPixmap(q_img.scaled(100, 100, Qt.KeepAspectRatio))
            self.table.setCellWidget(row, 0, preview_label)

            # Add filename and page number to the table
            self.table.setItem(row, 1, QTableWidgetItem(filename))
            self.table.setItem(row, 2, QTableWidgetItem(str(page_number)))

            # Add status (✅ or ❌)
            status_label = QLabel("✅" if re.match(r'\b\d+\b|\b[IVXLCDM]+\b|\b[ivxlcdm]+\b', page_number) else "❌")
            self.table.setCellWidget(row, 3, status_label)

        self.progress_label.setText("Processing Complete")
        self.table.doubleClicked.connect(self.show_image)
        self.book_result_btn.setEnabled(True)

        if detected_pages:
            try:
                # Convert Roman numerals to integers for sorting
                detected_pages_sorted = sorted(detected_pages, key=lambda x: self.roman_to_int(x) if isinstance(x, str) and x.isalpha() else int(x))
                expected_pages = list(range(1, self.roman_to_int(detected_pages_sorted[-1]) + 1 if isinstance(detected_pages_sorted[-1], str) else int(detected_pages_sorted[-1]) + 1))
                missing_pages = sorted(set(expected_pages) - set([self.roman_to_int(x) if isinstance(x, str) and x.isalpha() else int(x) for x in detected_pages_sorted]))
                
                if missing_pages:
                    self.status_bar.showMessage(f"Missing pages: {', '.join(map(str, missing_pages))}")
                else:
                    self.status_bar.showMessage("All pages accounted for!")
            except Exception as e:
                self.status_bar.showMessage(f"Error sorting pages: {str(e)}")

    def roman_to_int(self, s):
        roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        prev_value = 0
        for char in reversed(s):
            value = roman[char]
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
        return total

    def show_image(self, index):
        row = index.row()
        image_path = self.image_paths.get(row)
        if image_path:
            img = cv2.imread(image_path)
            if img is None:
                QMessageBox.critical(self, "Error", "Unable to load image!")
                return

            # Check if a page number was detected
            page_number = self.results.get(image_path, "")
            if page_number == "Page number found but not recognized" or page_number.isdigit() or re.match(r'\b[IVXLCDM]+\b', page_number):
                # Get the bounding box for the detected page number
                results = self.processing_thread.detector.model.predict(img, conf=0.5)
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    x1, y1, x2, y2 = map(int, max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1])))
                    img = img[y1:y2, x1:x2]

            # Save the cropped/full image to a temporary file
            temp_image_path = os.path.join(self.temp_dir, f"temp_{row}.jpg")
            cv2.imwrite(temp_image_path, img)

            # Display the image in the ImageViewer dialog
            viewer = ImageViewer(temp_image_path)
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
                    'detected_pages': [],
                    'missing_pages': [],
                    'all_pages_above_300dpi': True,  # Assume all pages are above 300 DPI initially
                    'in_order_pages': True,  # Assume pages are in order initially
                }

            if re.match(r'\b\d+\b|\b[IVXLCDM]+\b', page_number):
                book_results[book_name]['detected_pages'].append(page_number)
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

        # Calculate missing pages, check DPI, and verify page order for each book
        for book, details in book_results.items():
            detected_pages = details['detected_pages']
            if detected_pages:
                try:
                    # Convert Roman numerals to integers for sorting
                    detected_pages_sorted = sorted(detected_pages, key=lambda x: self.roman_to_int(x) if isinstance(x, str) and x.isalpha() else int(x))
                    expected_pages = list(range(1, self.roman_to_int(detected_pages_sorted[-1]) + 1 if isinstance(detected_pages_sorted[-1], str) else int(detected_pages_sorted[-1]) + 1))
                    missing_pages = sorted(set(expected_pages) - set([self.roman_to_int(x) if isinstance(x, str) and x.isalpha() else int(x) for x in detected_pages_sorted]))
                    details['missing_pages'] = missing_pages

                    # Check if all pages are above 300 DPI (placeholder logic)
                    # You can add logic to verify DPI from the image metadata
                    details['all_pages_above_300dpi'] = True  # Replace with actual DPI check

                    # Check if pages are in order
                    details['in_order_pages'] = detected_pages == sorted(detected_pages, key=lambda x: self.roman_to_int(x) if isinstance(x, str) and x.isalpha() else int(x))
                except Exception as e:
                    details['missing_pages'] = []
                    details['in_order_pages'] = False
                    print(f"Error processing book {book}: {str(e)}")

        return book_results

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())