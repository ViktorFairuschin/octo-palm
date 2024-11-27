# Copyright (c) 2024 Viktor Fairuschin


import typing
import pathlib
import cv2
import numpy as np


def load_image(filepath: typing.Union[str, pathlib.Path]) -> np.ndarray:
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def resize_image(image: np.ndarray, size: typing.Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    return image / 255.0


