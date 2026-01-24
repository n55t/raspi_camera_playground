from picamera2 import Picamera2
import cv2
import numpy as np
import ncnn
import yaml

from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.ops import xywh2xyxy


PARAM_PATH = "yolo11n_ncnn_model/model.ncnn.param"
BIN_PATH = "yolo11n_ncnn_model/model.ncnn.bin"
META_PATH = "yolo11n_ncnn_model/metadata.yaml"

INPUT_SIZE = 640
CONF_TH = 0.25
NMS_TH = 0.45


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def nms_xyxy(boxes, scores, thresh):
    if len(boxes) == 0:
        return []

    idxs = scores.argsort()[::-1]
    keep = []

    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area2 = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])

        iou = inter / (area1 + area2 - inter + 1e-6)
        idxs = idxs[1:][iou < thresh]

    return keep


class YoloNCNNDetector:
    def __init__(self, param_path, bin_path, meta_path):
        self.net = ncnn.Net()
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        with open(meta_path) as f:
            meta = yaml.safe_load(f)

        names_dict = meta["names"]
        self.class_names = [names_dict[i] for i in sorted(names_dict)]

    @staticmethod
    def _preprocess(frame):
        h, w, _ = frame.shape
        scale = INPUT_SIZE / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))

        resized = cv2.resize(frame, (nw, nh))
        padded = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        padded[:nh, :nw] = resized

        return padded, scale

    def infer(self, frame):
        orig_h, orig_w = frame.shape[:2]
        input_img, scale = self._preprocess(frame)

        mat = ncnn.Mat.from_pixels(
            input_img,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            INPUT_SIZE,
            INPUT_SIZE,
        )
        mat.substract_mean_normalize([], [1/255., 1/255., 1/255.])

        ex = self.net.create_extractor()
        ex.input("in0", mat)
        ret, out0 = ex.extract("out0")
        if ret != 0:
            return []

        raw = np.array(out0).astype(np.float32)
        if raw.ndim == 3:
            raw = raw[0]
        if raw.shape[0] == 84:
            pred = raw.T
        else:
            pred = raw

        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        cls_scores = pred[:, 4:]

        if cls_scores.max() > 1.5 or cls_scores.min() < -0.1:
            cls_scores = sigmoid(cls_scores)

        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(len(cls_ids)), cls_ids]

        mask = scores > CONF_TH
        cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]

        if len(scores) == 0:
            return []

        boxes_xywh = np.stack([cx, cy, bw, bh], axis=1)
        boxes_xyxy = xywh2xyxy(boxes_xywh)

        boxes_xyxy[:, [0, 2]] /= scale
        boxes_xyxy[:, [1, 3]] /= scale

        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w - 1)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h - 1)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w - 1)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h - 1)

        keep = nms_xyxy(boxes_xyxy, scores, NMS_TH)

        detections = []
        for i in keep:
            detections.append({
                "box": boxes_xyxy[i],
                "score": scores[i],
                "class_id": int(cls_ids[i]),
                "label": self.class_names[int(cls_ids[i])]
            })

        return detections


def draw_detections(frame, detections):
    annotator = Annotator(frame, line_width=2)

    for d in detections:
        box = d["box"]
        cls_id = d["class_id"]
        label = f'{d["label"]}:{d["score"]:.2f}'

        annotator.box_label(box, label, color=colors(cls_id, True))

    return annotator.result()


def main():
    detector = YoloNCNNDetector(PARAM_PATH, BIN_PATH, META_PATH)

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        detections = detector.infer(frame)
        frame = draw_detections(frame, detections)

        cv2.imshow("cam", frame)
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()
