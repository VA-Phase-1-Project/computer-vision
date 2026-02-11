import cv2
import time
from .service import FaceDetector, draw_faces

detector = FaceDetector(min_confidence=0.3)
current_face_count = 0


def mjpeg_stream_from_video_path(video_path: str):
    global current_face_count

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    frame_delay = 1.0 / fps

    while True:
        start_time = time.time()

        success, frame = cap.read()
        if not success:
            break

        detections = detector.detect(frame)
        current_face_count = len(detections)

        frame = draw_faces(frame, detections)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )

        elapsed = time.time() - start_time
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
