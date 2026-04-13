"""
FIT-VTON: Measurement-Conditioned Virtual Try-On

A lightweight FitAdapter on top of IDM-VTON that injects body and garment
measurement embeddings into cross-attention layers, enabling the first VTON
model that correctly renders how ill-fitting garments look on different body types.
"""

__version__ = "0.1.0"
__author__ = "NatalieCarlebach1"

from fit_vton.models.measurement_encoder import MeasurementEncoder
from fit_vton.models.fit_adapter import FitAdapter, FitAttnProcessor
from fit_vton.models.pipeline import FitVTONPipeline

__all__ = [
    "MeasurementEncoder",
    "FitAdapter",
    "FitAttnProcessor",
    "FitVTONPipeline",
]
