from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolo11n.pt")
def main():
    # Train the model
    results = model.train(
        data="yolov8n.yaml",
        epochs=100,
        imgsz=1024,
        batch=8,
        name="page_number_detector"
    )

if __name__=="__main__":
    main()