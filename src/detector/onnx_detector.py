# Copyright (c) 2024 Viktor Fairuschin


import typing
import pathlib

import onnxruntime
import numpy as np

from src.models import BaseDetector
from src.image_ops import resize_image, normalize_image


class YOLODetector(BaseDetector):

    def __init__(self, filepath: typing.Union[str, pathlib.Path], threshold: float = 0.5):
        self.threshold = threshold
        self.session = onnxruntime.InferenceSession(filepath, providers=["CPUExecutionProvider"])
        self.img_h, self.img_w, self.img_c = None, None, None

    def detect(self, image: np.ndarray):
        inputs = self.preprocess(image)
        outputs = self.session.run(None, input_feed={"images": inputs})
        results = self.postprocess(outputs[0])
        return results

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        self.img_h, self.img_w, self.img_c = image.shape

        b, c, w, h = self.session.get_inputs()[0].shape
        out = resize_image(image, size=(w, h))
        out = normalize_image(out)
        out = np.transpose(out, axes=(2, 0, 1))
        out = np.expand_dims(out, axis=0).astype(np.float32)
        return out

    def postprocess(self, inputs: np.ndarray) -> np.ndarray:
        out = np.squeeze(inputs)
        out = np.transpose(out)
        boxes = out[:, :4]
        scores = out[:, 4:]
        probs = np.max(scores, axis=1)
        classes = np.argmax(scores, axis=1)

        return boxes, probs, classes

