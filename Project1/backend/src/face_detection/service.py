from typing import List, Tuple

import cv2
import numpy as np
import mediapipe as mp

Box = Tuple[int, int, int, int]


class FaceDetector:
    """
    Uses MediaPipe SOLUTIONS (most stable).
    model_selection:
      0 = short-range (selfie / close faces)
      1 = long-range (farther faces)
    """
    def __init__(self, min_confidence: float = 0.3, model_selection: int = 1):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Your mediapipe install is broken (no mp.solutions). "
                "Run: pip uninstall -y mediapipe && pip install mediapipe"
            )

        self._fd = mp.solutions.face_detection.FaceDetection(
            model_selection=int(model_selection),
            min_detection_confidence=float(min_confidence),
        )

    def detect_boxes(self, frame_bgr: np.ndarray) -> List[Box]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]

        # MediaPipe solutions expects RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        results = self._fd.process(rgb)
        boxes: List[Box] = []

        if results.detections:
            for det in results.detections:
                b = det.location_data.relative_bounding_box
                x1 = int(b.xmin * w)
                y1 = int(b.ymin * h)
                x2 = int((b.xmin + b.width) * w)
                y2 = int((b.ymin + b.height) * h)

                # clamp
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))

        return boxes


def draw_faces(frame: np.ndarray, boxes: List[Box]) -> np.ndarray:
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame


# One global detector (fast + stable)
DETECTOR = FaceDetector(min_confidence=0.3, model_selection=1)
