r"""
All the model bases, which take as input the raw timeseries
and return a vector
"""
from .lstm import LSTM


STR2BASE = {"lstm": LSTM}


__all__ = ["STR2BASE"]
