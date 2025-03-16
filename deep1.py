import sys
import tempfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
                             QProgressBar, QFileDialog, QLabel, QMessageBox, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QPropertyAnimation, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QMovie
import cv2
from pdf2image import convert_from_path
from PIL.ImageQt import ImageQt
from PIL import Image
import os

# Your PageNumberDetector class
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


# PyQt5 GUI Application
class ProcessingThread(QThread):
    update_progress = pyqtSignal(int)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    dpi_warning = pyqtSignal(str)

    def __init__(self, folder_path, model_path):
        super().__init__()
        self.folder_path = folder_path
        self.model_path = model_path
        self.temp_dir = tempfile.TemporaryDirectory()
        self.detector = PageNumberDetector(model_path)

    def run(self):
        try:
            pdf_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.pdf')]
            total_pages = 0
            image_paths = []

            # Convert PDFs to images
            for pdf_file in pdf_files:
                pdf_path = os.path.join(self.folder_path, pdf_file)
                images = convert_from_path(pdf_path, dpi=300, output_folder=self.temp_dir.name,
                                          fmt='jpeg', paths_only=True)
                image_paths.extend(images)
                total_pages += len(images)

            # Process images
            results = {}
            for idx, img_path in enumerate(image_paths):
                self.update_progress.emit(int((idx + 1) / total_pages * 100))
                
                # Check DPI
                with Image.open(img_path) as img:
                    if img.info.get('dpi', (0, 0))[0] < 300:
                        self.dpi_warning.emit(os.path.basename(img_path))
                
                result = self.detector.detect_page_number(img_path, self.temp_dir.name)
                results[os.path.basename(img_path)] = result

            self.result_ready.emit(results)

        except Exception as e:
            self.error_occurred.emit(str(e))

class ImageViewer(QDialog):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Page Viewer")
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout()
        self.label = QLabel()
        pixmap = QPixmap(image_path)
        self.label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
        layout.addWidget(self.label)
        self.setLayout(layout)

class AnimatedTick(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress = 0
        self.animation = QPropertyAnimation(self, b"progress")
        self.animation.setDuration(1000)
        self.animation.setStartValue(0)
        self.animation.setEndValue(100)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(0, 255, 0, int(255 * self._progress / 100))
        painter.setPen(color)
        painter.setBrush(color)
        
        size = min(self.width(), self.height()) * 0.8
        x = (self.width() - size) / 2
        y = (self.height() - size) / 2
        rect = QRect(x, y, size, size)
        painter.drawEllipse(rect)
        
        # Draw checkmark
        painter.setPen(QColor(255, 255, 255))
        painter.drawLine(rect.left() + size*0.2, rect.center().y(),
                        rect.center().x(), rect.bottom() - size*0.2)
        painter.drawLine(rect.center().x(), rect.bottom() - size*0.2,
                        rect.right() - size*0.2, rect.top() + size*0.3)

    def get_progress(self):
        return self._progress

    def set_progress(self, value):
        self._progress = value
        self.update()

    progress = property(get_progress, set_progress)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Page Number Inspector")
        self.setGeometry(100, 100, 1024, 768)
        
        # Main Widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Controls
        controls_layout = QHBoxLayout()
        self.browse_btn = QPushButton("Browse Folder")
        self.browse_btn.clicked.connect(self.browse_folder)
        controls_layout.addWidget(self.browse_btn)
        
        self.progress_bar = QProgressBar()
        controls_layout.addWidget(self.progress_bar)
        layout.addLayout(controls_layout)
        
        # Results Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Thumbnail", "Filename", "Page Number", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(0, 150)
        self.table.doubleClicked.connect(self.show_image)
        layout.addWidget(self.table)
        
        # Status Bar
        self.status_bar = self.statusBar()
        
        self.processing_thread = None
        self.temp_images = {}

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select PDF Folder")
        if folder:
            self.process_folder(folder)

    def process_folder(self, folder_path):
        self.table.setRowCount(0)
        self.progress_bar.setValue(0)
        
        self.processing_thread = ProcessingThread(folder_path, "best.pt")
        self.processing_thread.update_progress.connect(self.progress_bar.setValue)
        self.processing_thread.result_ready.connect(self.show_results)
        self.processing_thread.error_occurred.connect(self.show_error)
        self.processing_thread.dpi_warning.connect(self.show_dpi_warning)
        self.processing_thread.start()

    def show_results(self, results):
        self.table.setRowCount(len(results))
        page_numbers = []
        
        for row, (filename, page_number) in enumerate(results.items()):
            # Thumbnail
            thumbnail_label = QLabel()
            pixmap = QPixmap(os.path.join(self.processing_thread.temp_dir.name, filename))
            thumbnail_label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))
            self.table.setCellWidget(row, 0, thumbnail_label)
            
            # Filename
            self.table.setItem(row, 1, QTableWidgetItem(filename))
            
            # Page Number
            page_item = QTableWidgetItem(str(page_number))
            self.table.setItem(row, 2, page_item)
            
            # Status
            status_widget = QWidget()
            layout = QHBoxLayout(status_widget)
            layout.setAlignment(Qt.AlignCenter)
            
            if page_number.isdigit():
                page_numbers.append(int(page_number))
                anim = AnimatedTick()
                anim.animation.start()
                layout.addWidget(anim)
            else:
                error_label = QLabel("❌")
                layout.addWidget(error_label)
            
            self.table.setCellWidget(row, 3, status_widget)
        
        # Check missing pages
        if page_numbers:
            sorted_pages = sorted(page_numbers)
            expected_pages = list(range(1, sorted_pages[-1] + 1))
            missing_pages = set(expected_pages) - set(sorted_pages)
            
            if missing_pages:
                self.status_bar.showMessage(f"Missing pages: {', '.join(map(str, missing_pages))}")
            else:
                self.status_bar.showMessage("All pages accounted for!")

    def show_image(self, index):
        row = index.row()
        filename = self.table.item(row, 1).text()
        image_path = os.path.join(self.processing_thread.temp_dir.name, filename)
        viewer = ImageViewer(image_path)
        viewer.exec_()

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    def show_dpi_warning(self, filename):
        self.status_bar.showMessage(f"Low DPI warning for {filename}!", 5000)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())