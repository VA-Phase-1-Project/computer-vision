import cv2
from .service import DETECTOR, draw_faces

counts = {"faces": 0}

def get_counts():
    return counts

def mjpeg_stream_from_video_path(path):
    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = DETECTOR.detect_boxes(frame)
        counts["faces"] = len(boxes)

        draw_faces(frame, boxes)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()
