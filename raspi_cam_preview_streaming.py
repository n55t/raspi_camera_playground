from flask import Flask, Response
from picamera2 import Picamera2
import cv2

import settings

picam = Picamera2()
if settings.CAPTURE_SIZE:
    picam.configure(
        picam.create_video_configuration(main={"size": settings.CAPTURE_SIZE})
    )
else:
    picam.configure(picam.create_video_configuration())
picam.start()
picam.set_controls(
    {"AfMode": 2 if settings.AUTO_FOCUS_ENABLED else 0}
)

app = Flask(__name__)


def gen():
    while True:
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if settings.STREAM_SIZE:
            frame = cv2.resize(frame, settings.STREAM_SIZE)
        _, jpeg = cv2.imencode(settings.JPEG_FORMAT, frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )


@app.route(settings.VIDEO_ROUTE)
def video():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


app.run(host=settings.HOST, port=settings.PORT)
