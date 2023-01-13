import sys
import argparse

sys.path.append("./datasets")
from datasets.datamodule import CustomDataModule
from system import CustomSystem

from logs.config_handler import ConfigHandler


def option_parser():
    parser = argparse.ArgumentParser()

    return parser.parse_args()


def main():
    args = option_parser()

    dataset = CustomDataModule()
    system = CustomSystem({
        "model": "model",
        "model_name": "CustomModel",
        "loss": "mae",
        "loss_name": "MeanAbsoluteError",
    })
    recorder = ConfigHandler(dataset, system)
    recorder.record()


if __name__ == '__main__':
    main()
