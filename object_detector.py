from ultralytics import YOLO
import torch
import cv2

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def detect_objects(self, frame):
        results = self.model(frame)
        return results[0]