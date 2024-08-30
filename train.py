from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' as the starting point

# Train the model on your custom dataset
model.train(data="/Users/harrison/Documents/RND/modelId/datasets/dataset.yaml", epochs=100, imgsz=640)  # Adjust parameters as needed

# Save the trained model
model.save("yoloImageDetect.pt")