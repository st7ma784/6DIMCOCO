"""Loss functions for multi-dimensional CLIP training."""

from .base_loss import BaseLoss, LossRegistry
from .ndim_losses import (
    EinsumLoss,
    EuclideanDistanceLoss,
    NormBasedLoss,
    CosineSimilarityLoss,
    StockCLIPLoss
)
from .loss_factory import create_loss_function, get_available_losses

__all__ = [
    'BaseLoss',
    'LossRegistry', 
    'EinsumLoss',
    'EuclideanDistanceLoss',
    'NormBasedLoss',
    'CosineSimilarityLoss',
    'StockCLIPLoss',
    'create_loss_function',
    'get_available_losses'
]
