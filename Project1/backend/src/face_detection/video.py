import time
import cv2
import numpy as np
from typing import List, Tuple

from .service import DETECTOR, draw_faces

Box = Tuple[int, int, int, int]

# Stable defaults
TARGET_WIDTH = 960           # keeps quality; reduces load
FPS_LIMIT = 25
NEW_FACE_DISTANCE = 80

current_face_count = 0
total_unique_faces = 0
is_active = False
last_error = None


def get_counts():
    return {
        "current_face_count": int(current_face_count),
        "total_unique_faces": int(total_unique_faces),
        "is_active": bool(is_active),
        "last_error": last_error,
    }


def resize_keep_ratio(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / float(w)
    return cv2.resize(frame, (target_width, int(h * scale)))


def mjpeg_stream_from_video_path(video_path: str):
    global current_face_count, total_unique_faces, is_active, last_error

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        last_error = "Cannot open video"
        is_active = False
        raise RuntimeError(last_error)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(src_fps) if src_fps and src_fps > 0 else FPS_LIMIT
    fps = min(fps, FPS_LIMIT)
    frame_time = 1.0 / max(1, fps)

    current_face_count = 0
    total_unique_faces = 0
    is_active = True
    last_error = None

    known_centroids: List[Tuple[int, int]] = []

    try:
        last_tick = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = resize_keep_ratio(frame, TARGET_WIDTH)

            boxes = DETECTOR.detect_boxes(frame)
            current_face_count = len(boxes)

            # Unique count by centroid distance (simple + stable)
            for (x1, y1, x2, y2) in boxes:
                c = ((x1 + x2) // 2, (y1 + y2) // 2)
                is_new = True
                for k in known_centroids:
                    if np.linalg.norm(np.array(c) - np.array(k)) < NEW_FACE_DISTANCE:
                        is_new = False
                        break
                if is_new:
                    known_centroids.append(c)
                    total_unique_faces += 1

            # draw + overlay
            draw_faces(frame, boxes)
            cv2.putText(frame, f"Live: {current_face_count}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            cv2.putText(frame, f"Unique: {total_unique_faces}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )

            # FPS control
            now = time.time()
            elapsed = now - last_tick
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last_tick = time.time()

    except Exception as e:
        last_error = str(e)
        raise
    finally:
        cap.release()
        is_active = False
