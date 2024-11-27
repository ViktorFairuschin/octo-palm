# Copyright (c) 2024 Viktor Fairuschin


import typing
import pathlib
import yaml


def load_config(filepath: typing.Union[str, pathlib.Path]) -> dict:
    with open(filepath, "r") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return params

#