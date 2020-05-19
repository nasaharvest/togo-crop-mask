import sys
from pathlib import Path

sys.path.append("..")

from src.engineer import (
    GeoWikiEngineer,
    TogoEngineer,
    TogoEvaluationEngineer,
)


def engineer_geowiki():
    engineer = GeoWikiEngineer(Path("../data"))
    engineer.engineer(val_set_size=0.2, test_set_size=0)


def engineer_togo():
    engineer = TogoEngineer(Path("../data"))
    engineer.engineer(val_set_size=0.2, test_set_size=0)

    eval_engineer = TogoEvaluationEngineer(Path("../data"))
    eval_engineer.engineer()


if __name__ == "__main__":
    engineer_geowiki()
    engineer_togo()
