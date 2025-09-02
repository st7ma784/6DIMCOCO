"""Base classes for loss functions in the 6DIMCOCO framework."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class LossRegistry:
    """Registry for loss functions to enable dynamic creation."""
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a loss function."""
        def decorator(loss_class):
            cls._registry[name] = loss_class
            return loss_class
        return decorator
    
    @classmethod
    def get_loss_class(cls, name: str) -> type:
        """Get a loss class by name."""
        if name not in cls._registry:
            raise ValueError(f"Loss function '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name]
    
    @classmethod
    def list_available(cls) -> list:
        """List all available loss functions."""
        return list(cls._registry.keys())


class BaseLoss(nn.Module, ABC):
    """Abstract base class for all loss functions."""
    
    def __init__(self, 
                 normalize: bool = True,
                 log_variance: bool = False,
                 numerical_stability_eps: float = 1e-8,
                 **kwargs):
        """
        Initialize base loss function.
        
        Args:
            normalize: Whether to normalize input features
            log_variance: Whether to apply log variance
            numerical_stability_eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.normalize = normalize
        self.log_variance = log_variance
        self.eps = numerical_stability_eps
        
    @abstractmethod
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Compute loss from input features.
        
        Args:
            *features: Variable number of feature tensors
            
        Returns:
            Loss tensor
        """
        pass
    
    def _normalize_features(self, *features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Normalize features if required."""
        if not self.normalize:
            return features
        
        normalized = []
        for feat in features:
            norm = feat.norm(dim=-1, keepdim=True)
            # Add epsilon for numerical stability
            norm = torch.clamp(norm, min=self.eps)
            normalized.append(feat / norm)
        
        return tuple(normalized)
    
    def _check_for_anomalies(self, tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """Check for NaN or Inf values and handle them."""
        if torch.isnan(tensor).any():
            logger.warning(f"NaN detected in {name}")
            tensor = torch.nan_to_num(tensor, nan=0.0)
        
        if torch.isinf(tensor).any():
            logger.warning(f"Inf detected in {name}")
            tensor = torch.nan_to_num(tensor, posinf=1e6, neginf=-1e6)
        
        return tensor
    
    def _validate_inputs(self, *features: torch.Tensor) -> None:
        """Validate input tensors."""
        if not features:
            raise ValueError("At least one feature tensor must be provided")
        
        # Check that all features have the same batch size
        batch_sizes = [feat.shape[0] for feat in features]
        if len(set(batch_sizes)) > 1:
            raise ValueError(f"All features must have the same batch size. Got: {batch_sizes}")
        
        # Check for empty tensors
        for i, feat in enumerate(features):
            if feat.numel() == 0:
                raise ValueError(f"Feature {i} is empty")
    
    def compute_loss_with_validation(self, *features: torch.Tensor) -> torch.Tensor:
        """Compute loss with input validation and anomaly detection."""
        self._validate_inputs(*features)
        
        # Normalize if required
        if self.normalize:
            features = self._normalize_features(*features)
        
        # Compute loss
        loss = self.forward(*features)
        
        # Check for anomalies
        loss = self._check_for_anomalies(loss, "loss")
        
        return loss


class MultiDimensionalLoss(BaseLoss):
    """Base class for multi-dimensional loss functions."""
    
    def __init__(self, dimensions: int = 6, **kwargs):
        """
        Initialize multi-dimensional loss.
        
        Args:
            dimensions: Number of dimensions for the loss computation
        """
        super().__init__(**kwargs)
        self.dimensions = dimensions
        
        if dimensions < 2:
            raise ValueError("Dimensions must be at least 2")
    
    def _expand_for_dimensions(self, *features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Expand features for multi-dimensional computation."""
        expanded = []
        
        for i, feat in enumerate(features):
            # Create the appropriate view for this feature in the n-dimensional space
            view_shape = [1] * len(features)
            view_shape[i] = feat.shape[0]
            view_shape.append(feat.shape[-1])  # Feature dimension
            
            expanded_feat = feat.view(*view_shape)
            expanded.append(expanded_feat)
        
        return tuple(expanded)


class StabilizedLoss(BaseLoss):
    """Loss with enhanced numerical stability."""
    
    def __init__(self, 
                 gradient_clip_val: Optional[float] = None,
                 loss_scale: float = 1.0,
                 **kwargs):
        """
        Initialize stabilized loss.
        
        Args:
            gradient_clip_val: Value for gradient clipping
            loss_scale: Scale factor for loss
        """
        super().__init__(**kwargs)
        self.gradient_clip_val = gradient_clip_val
        self.loss_scale = loss_scale
    
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """Forward pass with stability checks."""
        loss = self._compute_raw_loss(*features)
        
        # Apply loss scaling
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale
        
        # Gradient clipping if specified
        if self.gradient_clip_val is not None and loss.requires_grad:
            torch.nn.utils.clip_grad_value_(loss, self.gradient_clip_val)
        
        return loss
    
    @abstractmethod
    def _compute_raw_loss(self, *features: torch.Tensor) -> torch.Tensor:
        """Compute the raw loss without stability modifications."""
        pass
