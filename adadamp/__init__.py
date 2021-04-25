__version__ = "0.1.11"

from .dampers import (
    BaseDamper,
    GeoDamp
)
from ._dist import DaskBaseDamper, DaskClassifier
from .utils import _get_resnet18

__all__ = [
    "BaseDamper",
    "GeoDamp",
]
