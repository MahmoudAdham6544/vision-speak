import cv2
from object_detector import ObjectDetector
from caption_generator import CaptionGenerator
from utils import crop_box

def main():
    detector = ObjectDetector()
    captioner = CaptionGenerator()

    cap = cv2.VideoCapture(0)
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 != 0:
            results = detector.detect_objects(frame)
            
            for box in results.boxes:
                conf = box.conf[0].item()
                if conf < 0.2:
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                crop = crop_box(frame, (x1, y1, x2, y2))

                # try:
                caption = captioner.generate_caption(crop)
                # except Exception as e:
                #     caption = "Caption failed"

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, caption, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("VisionSpeak", frame)

            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()