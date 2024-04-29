from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='C:/Users/houss/Desktop/projects/imageclassificationyolov8/dataset2', epochs=1, imgsz=64)