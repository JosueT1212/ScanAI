import numpy as np
from shelf.config import Config
from shelf.detect import StubDetector
from shelf.fuse import Detection


def test_stub_detector_returns_a_centred_box():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = StubDetector(Config()).infer(frame)
    assert len(dets) == 1
    d = dets[0]
    assert isinstance(d, Detection)
    assert (d.x1 + d.x2) / 2 == 320.0
    assert d.conf >= 0.5


def test_stub_detector_scales_to_frame_width():
    frame = np.zeros((240, 1000, 3), dtype=np.uint8)
    d = StubDetector(Config()).infer(frame)[0]
    assert (d.x1 + d.x2) / 2 == 500.0
