from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yoloImageDetect.pt")

# Export the model to TFLite format  pip install onnx2tf

model.export(format="tflite")