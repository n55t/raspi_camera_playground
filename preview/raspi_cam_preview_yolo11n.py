from picamera2 import Picamera2
from ultralytics import YOLO
import cv2

MODEL_PATH = "yolo11n.pt"


def main() -> None:
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()

    # YOLO11 default model (nano)
    model = YOLO(MODEL_PATH)

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("cam", annotated)
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()
