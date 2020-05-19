import sys
from argparse import ArgumentParser

sys.path.append("..")

from src.models import STR2MODEL, train_model


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)

    model_args = STR2MODEL["land_cover"].add_model_specific_args(parser).parse_args()
    model = STR2MODEL["land_cover"](model_args)

    train_model(model, model_args)
