"""Factory for creating loss functions dynamically."""

import torch
from typing import Dict, Any, Optional
from .base_loss import BaseLoss, LossRegistry
from ..config.base_config import ModelConfig


def create_loss_function(
    loss_type: str,
    config: Optional[ModelConfig] = None,
    **kwargs
) -> BaseLoss:
    """
    Create a loss function based on type and configuration.
    
    Args:
        loss_type: Type of loss function to create
        config: Model configuration object
        **kwargs: Additional arguments for loss function
        
    Returns:
        Initialized loss function
        
    Raises:
        ValueError: If loss_type is not recognized
    """
    # Get the loss class from registry
    loss_class = LossRegistry.get_loss_class(loss_type)
    
    # Prepare arguments from config if provided
    loss_kwargs = {}
    if config is not None:
        loss_kwargs.update({
            'normalize': config.normalize_logits,
            'log_variance': config.log_variance,
            'dimensions': int(config.dimensions) if config.dimensions > 0 else 6
        })
    
    # Override with any explicitly provided kwargs
    loss_kwargs.update(kwargs)
    
    return loss_class(**loss_kwargs)


def get_available_losses() -> Dict[str, str]:
    """
    Get all available loss functions with descriptions.
    
    Returns:
        Dictionary mapping loss names to descriptions
    """
    descriptions = {
        'stock_clip': 'Standard CLIP contrastive loss',
        'einsum': 'Einstein summation based n-dimensional loss',
        'euclidean_distance': 'Euclidean distance based loss with stability',
        'norm_based': 'Norm-based loss with multiple variants',
        'cosine_similarity': 'Cosine similarity based multi-dimensional loss'
    }
    
    available = LossRegistry.list_available()
    return {name: descriptions.get(name, 'No description available') for name in available}


def create_loss_from_legacy_version(version: int, **kwargs) -> BaseLoss:
    """
    Create loss function based on legacy version numbers.
    
    Args:
        version: Legacy loss version number (0-18)
        **kwargs: Additional arguments
        
    Returns:
        Appropriate loss function
    """
    # Map legacy versions to new loss types
    version_map = {
        0: ('einsum', {}),
        1: ('euclidean_distance', {'use_sqrt': False}),
        2: ('euclidean_distance', {'use_sqrt': True}),
        3: ('euclidean_distance', {'variant': 'v3'}),
        4: ('euclidean_distance', {'variant': 'v4'}),
        5: ('euclidean_distance', {'variant': 'v5'}),
        6: ('euclidean_distance', {'variant': 'v6'}),
        7: ('norm_based', {'variant': 'v1'}),
        8: ('norm_based', {'variant': 'v2'}),
        9: ('norm_based', {'variant': 'v3'}),
        10: ('norm_based', {'variant': 'v4'}),
        11: ('norm_based', {'variant': 'v5'}),
        12: ('norm_based', {'variant': 'v5', 'normalize': False}),
        13: ('euclidean_distance', {'variant': 'v7'}),
        14: ('euclidean_distance', {'variant': 'v8'}),
        15: ('norm_based', {'variant': 'v6'}),
        16: ('norm_based', {'variant': 'v7'}),
        17: ('norm_based', {'variant': 'v8'}),
        18: ('norm_based', {'variant': 'v9'}),
    }
    
    if version not in version_map:
        raise ValueError(f"Unknown legacy version: {version}. Available: {list(version_map.keys())}")
    
    loss_type, loss_kwargs = version_map[version]
    loss_kwargs.update(kwargs)
    
    return create_loss_function(loss_type, **loss_kwargs)
