import cv2
from PIL import Image

def crop_box(frame, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    crop = frame[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))