"""Detector construction shared by training and diagnostic experiments."""

from .baselines import MSP, ODIN, OE, Mahalanobis
from .gnnsafe import GNNSafe

DETECTORS = {"msp": MSP, "gnnsafe": GNNSafe, "OE": OE, "ODIN": ODIN, "Mahalanobis": Mahalanobis}


def build_detector(features, classes, args, device):
    try:
        constructor = DETECTORS[args.method]
    except KeyError:
        raise ValueError(
            f"Unknown method {args.method!r}; choose from {', '.join(DETECTORS)}"
        ) from None
    return constructor(features, classes, args).to(device)
