YOLOv8 Training (Detector)

Demo (synthetic):

# Uses ultralytics CLI; ensure `pip install ultralytics`
yolo task=detect mode=train model=yolov8n.pt data=detector/data.yaml epochs=1 imgsz=640 batch=8
# Validation
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=detector/data.yaml
# Export (optional)
yolo mode=export model=runs/detect/train/weights/best.pt format=onnx


Real datasets:

Replace data/synth_det with PlantDoc annotations/images.

For segmentation masks, use yolov8n-seg.pt and segmentation labels (task=segment).