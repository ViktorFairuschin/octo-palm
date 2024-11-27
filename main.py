# Copyright (c) 2024 Viktor Fairuschin


import argparse

from src.utils import load_config
from src.image_ops import load_image
from src.detector import YOLODetector


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add description.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    return parser


def main(args: argparse.Namespace):
    config = load_config(args.config)

    image = load_image("image.jpg")

    detector_config = config["detector"]
    detector_filepath = detector_config.get("filepath")
    detector = YOLODetector(filepath=detector_filepath)
    boxes, probs = detector.detect(image)
    print(boxes.shape, probs.shape)
    print(boxes[0], max(probs[0]))


if __name__ == "__main__":
    main(create_parser().parse_args())

