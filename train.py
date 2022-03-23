# Shut Future Warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import typer

from utils.model import train_many_models
from utils.config import load_config


def main(
        path_data: str = "data-prep",
        path_output: str = "outputs",
        path_config: str = "/Desktop/nilm-thresholding-master/nilmth/config.toml",
):
    print(f"\nLoading config file from {path_config}")
    # Load config file
    config = load_config(path_config, "model")
    print("Done\n")

    # Run main results
    print(f"{config['model']}\n")

    train_many_models(path_data, path_output, config)


if __name__ == "__main__":
    typer.run(main)
