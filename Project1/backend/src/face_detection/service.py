import cv2
import numpy as np
import mediapipe as mp


class FaceDetector:

    def __init__(self, min_confidence: float = 0.1):
        self.min_confidence = min_confidence
        self._mp_face = mp.solutions.face_detection
        self.detector = self._mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.1,
        )

    def detect(self, frame: np.ndarray):

        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        detections = []

        if results.detections:
            for det in results.detections:
                score = float(det.score[0])
                if score < self.min_confidence:
                    continue

                bbox = det.location_data.relative_bounding_box

                x1 = int(max(0, bbox.xmin * w))
                y1 = int(max(0, bbox.ymin * h))
                x2 = int(min(w, (bbox.xmin + bbox.width) * w))
                y2 = int(min(h, (bbox.ymin + bbox.height) * h))

                detections.append(((x1, y1, x2, y2), score))

        return detections


def draw_faces(frame: np.ndarray, detections):

    output = frame.copy()

    for (x1, y1, x2, y2), score in detections:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{score:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        output,
        f"Faces: {len(detections)}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    return output
