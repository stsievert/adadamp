__version__ = "0.2.0rc8"

from .damping import (
    BaseDamper,
    AdaDamp,
    RadaDamp,
    PadaDamp,
    PrAdaDamp,
    GeoDamp,
    GeoDampLR,
    CntsDampLR,
    GradientDescent,
    ConvergenceError,
    AdaDampNN,
)
from ._dist import DaskBaseDamper, DaskClassifier
from .utils import _get_resnet18

__all__ = [
    "BaseDamper",
    "AdaDamp",
    "PadaDamp",
    "PrAdaDamp",
    "GeoDamp",
    "GeoDampLR",
    "CntsDampLR",
    "GradientDescent",
    "RadaDamp",
    "AdaDampNN",
]
