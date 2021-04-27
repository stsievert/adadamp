__version__ = "0.1.11"

from .damping import (
    BaseDamper,
    AdaDamp,
    RadaDamp,
    PadaDamp,
    GeoDamp,
    GeoDampLR,
    CntsDampLR,
    GradientDescent,
    ConvergenceError,
)
from ._dist import DaskBaseDamper, DaskClassifier
from .utils import _get_resnet18

__all__ = [
    "BaseDamper",
    "AdaDamp",
    "PadaDamp",
    "GeoDamp",
    "GeoDampLR",
    "CntsDampLR",
    "GradientDescent",
]
