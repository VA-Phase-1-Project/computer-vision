import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def detect_boxes(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)
        boxes = []
        if result.detections:
            for d in result.detections:
                bbox = d.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                boxes.append((x, y, width, height))
        return boxes

def draw_faces(frame, boxes):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

DETECTOR = FaceDetector()
