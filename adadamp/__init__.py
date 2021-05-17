__version__ = "0.2.0rc4"

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
    "RadaDamp",
]
