"""Object identity.

YOLOv8n knows 80 COCO classes. "Jumper wires" is not among them; laptop, cup,
bottle, book, scissors, keyboard, mouse, backpack and chair are. v1 tracks what
the detector can actually find rather than pretending to recognise more.
"""
from .fuse import Detection


class StubDetector:
    """One centred box. Lets the whole pipeline run without weights."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def infer(self, frame) -> list[Detection]:
        h, w = frame.shape[:2]
        bw, bh = w * 0.15, h * 0.25
        return [Detection(cls_name="cup", conf=0.88,
                          x1=w / 2 - bw / 2, y1=h / 2 - bh / 2,
                          x2=w / 2 + bw / 2, y2=h / 2 + bh / 2)]


class Detector:
    """ultralytics YOLOv8n. Loaded lazily so importing this module is cheap."""

    def __init__(self, cfg, weights: str = "yolov8n.pt") -> None:
        self.cfg = cfg
        self.weights = weights
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        return self._model

    def infer(self, frame) -> list[Detection]:
        model = self._load()
        out = []
        for r in model.predict(frame, verbose=False, conf=self.cfg.conf_floor):
            names = r.names
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                out.append(Detection(cls_name=names[int(b.cls[0])],
                                     conf=float(b.conf[0]),
                                     x1=x1, y1=y1, x2=x2, y2=y2))
        return out
