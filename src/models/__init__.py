from .models import LandCoverMapper
from .train_funcs import train_model


STR2MODEL = {
    "land_cover": LandCoverMapper,
}


__all__ = [
    "STR2MODEL",
    "LandCoverMapper",
    "train_model",
]
