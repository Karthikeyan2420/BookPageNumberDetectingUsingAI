# 📘 Book Page Number Detection

## 🚀 Project Overview
This AI-powered application detects and verifies page numbers in scanned book images. It ensures that:
- **Page numbers are detected correctly.**
- **DPI is checked and maintained at 300 DPI.**
- **Missing or out-of-sequence pages are flagged for review.**

## 🎯 Features
✅ **AI-Based Page Number Detection** – Uses a trained model to recognize page numbers.
✅ **DPI Validation** – Ensures scanned pages maintain a minimum of 300 DPI for clarity.
✅ **Page Sequence Checking** – Detects missing or misplaced pages.
✅ **Animated Visual Indicators** – Shows a tick ✅ for correct pages and highlights issues.
✅ **User-Friendly PyQt Interface** – Allows easy navigation and review of results.
✅ **Batch Processing** – Processes multiple PDFs in a folder at once.
✅ **Final Report Generation** – Displays:
   - Total Page Count 📄
   - Missing Page Numbers ❌
   - Lowest DPI Page Numbers ⚠️
   - Sequential Missing Pages 🔄

## 🛠️ Tech Stack
- **Python** 🐍
- **PyQt5** – GUI for user interaction
- **OpenCV** – Image processing
- **TensorFlow/Keras** – AI model for page detection
- **PDF2Image** – Converts PDFs to images
- **SQLite** – Stores processing results

## 📂 Project Structure
```
📦 book-page-detector
 ┣ 📂 data                # Sample book pages
 ┣ 📂 models              # Trained AI models
 ┣ 📂 ui                  # PyQt5 GUI components
 ┣ 📜 main.py             # Entry point for the application
 ┣ 📜 requirements.txt    # Dependencies
 ┗ 📜 README.md           # Project documentation
```

## 🚀 Installation & Setup
1. **Clone the Repository**
   ```sh
   git clone https://github.com/Karthikeyan2420/BookPageNumberDetectingUsingAI
   cd BookPageNumberDetectingUsingAI
   ```
2. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the Application**
   ```sh
   python PageDetection.py
   ```

## 📸 Screenshots
🔹 **Main Interface**
![Main UI](assets/main_ui.png)

🔹 **Page Detection Results**
![Results](assets/results.png)

## 🤝 Contributors
👨‍💻 **KARTHIKEYAN.KA** – Lead Developer  
📧 kakarthikeyan7670@gmail.com




