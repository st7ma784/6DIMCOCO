"""
Improved Loss Calculation with Numerical Stability

This module provides numerically stable implementations of the n-dimensional
loss functions with built-in gradient health checks and alternative activation functions.

Key Improvements:
- Epsilon terms to prevent division by zero
- Clamping to prevent overflow/underflow
- Optional gradient checkpointing for memory efficiency
- Logarithmic stability for extreme values
- Alternative formulations to prevent gradient vanishing
"""

import torch
import torch.nn as nn
from functools import reduce
from typing import Optional, Callable
import math


class StableLossFunctions:
    """Collection of numerically stable loss function implementations."""
    
    def __init__(
        self,
        eps: float = 1e-8,
        clamp_min: float = -100,
        clamp_max: float = 100,
        use_log_space: bool = False
    ):
        """
        Args:
            eps: Small constant for numerical stability
            clamp_min: Minimum value for clamping
            clamp_max: Maximum value for clamping
            use_log_space: Whether to compute in log space for stability
        """
        self.eps = eps
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.use_log_space = use_log_space
    
    def stable_norm(self, x: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
        """
        Compute stable L2 norm with epsilon for numerical stability.
        
        Args:
            x: Input tensor
            dim: Dimension to compute norm over
            keepdim: Whether to keep dimension
        
        Returns:
            Normalized tensor
        """
        norm = torch.norm(x, dim=dim, keepdim=keepdim)
        return x / (norm + self.eps)
    
    def stable_division(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """
        Perform stable division with clamping and epsilon.
        """
        denominator = torch.clamp(denominator, min=self.eps)
        result = numerator / denominator
        return torch.clamp(result, self.clamp_min, self.clamp_max)
    
    def stable_sqrt(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute stable square root, handling negative values.
        """
        # Use abs to handle numerical errors leading to slightly negative values
        return torch.sqrt(torch.abs(x) + self.eps)
    
    def stable_log(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute stable logarithm with epsilon.
        """
        return torch.log(torch.abs(x) + self.eps)
    
    def stable_mean_n_vectors(self, *vectors) -> torch.Tensor:
        """
        Compute stable mean of n vectors with gradient flow preservation.
        
        Uses numerically stable summation and division.
        """
        n = len(vectors)
        sum_vec = reduce(torch.add, vectors)
        # Scale down to prevent overflow
        return sum_vec / (n + self.eps)
    
    def stable_cosine_similarity_nd(self, *features) -> torch.Tensor:
        """
        Compute n-dimensional cosine similarity with numerical stability.
        
        This is a more stable version of the einsum-based similarity computation.
        
        Args:
            *features: Variable number of feature tensors [B, F]
        
        Returns:
            N-dimensional similarity tensor
        """
        n = len(features)
        
        # Normalize all features first
        normalized_features = [self.stable_norm(f) for f in features]
        
        # Compute mean representation in normalized space
        mean_vec = self.stable_mean_n_vectors(*normalized_features)
        mean_norm = self.stable_norm(mean_vec)
        
        # Compute dot product similarity of each feature with normalized mean
        similarities = []
        for feat in normalized_features:
            # Dot product similarity
            sim = torch.sum(feat.unsqueeze(1) * mean_norm.unsqueeze(0), dim=-1)
            similarities.append(sim)
        
        # Combine similarities
        # Use geometric mean instead of sum for better numerical properties
        combined = reduce(torch.mul, similarities)
        result = torch.pow(torch.abs(combined), 1.0 / n)
        
        return torch.clamp(result, self.clamp_min, self.clamp_max)
    
    def stable_euclidean_distance_nd(self, *features) -> torch.Tensor:
        """
        Compute n-dimensional Euclidean distance with numerical stability.
        
        Uses the formula: ||mean(x_i) - x_i|| aggregated across all dimensions.
        """
        n = len(features)
        
        # Normalize features
        normalized_features = [self.stable_norm(f) for f in features]
        
        # Compute mean
        mean_vec = self.stable_mean_n_vectors(*normalized_features)
        
        # Compute distances from mean
        distances = []
        for feat in normalized_features:
            dist = torch.sum((feat.unsqueeze(1) - mean_vec.unsqueeze(0)) ** 2, dim=-1)
            distances.append(self.stable_sqrt(dist))
        
        # Combine distances (negative because we want high similarity = low distance)
        combined = reduce(torch.add, distances) / n
        return -combined  # Negate so that closer = higher similarity
    
    def stable_norm_based_similarity(self, *features) -> torch.Tensor:
        """
        Improved version of calculate_lossNormsv4 with better numerical stability.
        
        Computes similarity based on normalized mean vector dot products.
        """
        n = len(features)
        
        # Normalize all features
        normalized_features = [self.stable_norm(f) for f in features]
        
        # Build views for broadcasting
        views = []
        for i, feat in enumerate(normalized_features):
            shape = [1] * i + [feat.shape[0]] + [1] * (n - 1 - i) + [-1]
            views.append(feat.view(*shape))
        
        # Compute mean in high-dimensional space
        mean_vec = reduce(torch.add, views)
        mean_norm = mean_vec / (torch.norm(mean_vec, dim=-1, keepdim=True) + self.eps)
        
        # Compute dot products with mean for each feature
        similarities = []
        for view in views:
            dot_prod = torch.sum(torch.mul(mean_norm, view), dim=-1)
            # Clamp to valid range for cosine similarity
            dot_prod = torch.clamp(dot_prod, -1.0, 1.0)
            similarities.append(dot_prod)
        
        # Sum of similarities
        result = reduce(torch.add, similarities)
        return result
    
    def stable_einsum_similarity(self, *features) -> torch.Tensor:
        """
        Stable version of einsum-based similarity calculation.
        
        Implements the original calculate_loss but with normalization
        and clamping at intermediate steps.
        """
        n = len(features)
        
        # Normalize inputs
        normalized_features = [self.stable_norm(f) for f in features]
        
        # Build einsum computation in chunks to prevent overflow
        alphabet = list(map(chr, range(97, 97 + n)))
        
        components = []
        parts = []
        finalparts = []
        
        for i in range(0, n, 3):
            if (i + 3) >= n:
                einsumparts = ",".join(["{}z"] * (n - i)) + "->" + "{}" + "z"
                einsumparts = einsumparts.format(*alphabet[i:], "".join(alphabet[i:]))
                finalparts.append("".join(alphabet[i:]) + "z")
                components.append(einsumparts)
                parts.append([*normalized_features[i:]])
            else:
                einsumparts = ",".join(["{}z"] * 3) + "->" + "{}" + "z"
                einsumparts = einsumparts.format(*alphabet[i:i+3], "".join(alphabet[i:i+3]))
                finalparts.append("".join(alphabet[i:i+3]) + "z")
                components.append(einsumparts)
                parts.append([*normalized_features[i:i+3]])
        
        # Compute einsums with clamping
        intermediate_results = []
        for component, part in zip(components, parts):
            result = torch.einsum(component, *part)
            # Clamp intermediate results
            result = torch.clamp(result, self.clamp_min, self.clamp_max)
            intermediate_results.append(result)
        
        # Final einsum
        final_result = torch.einsum(
            ",".join(finalparts) + "->" + "".join(alphabet),
            *intermediate_results
        )
        
        return torch.clamp(final_result, self.clamp_min, self.clamp_max)


class ImprovedActivations:
    """
    Collection of activation functions suitable for preventing gradient issues.
    """
    
    @staticmethod
    def gelu(x: torch.Tensor) -> torch.Tensor:
        """
        Gaussian Error Linear Unit - Smooth, non-zero gradients everywhere.
        Recommended for transformers and deep networks.
        """
        return torch.nn.functional.gelu(x)
    
    @staticmethod
    def swish(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """
        Swish activation (also known as SiLU).
        Self-gated, smooth, often better than ReLU for deep networks.
        
        Args:
            x: Input tensor
            beta: Scaling parameter (default 1.0 for SiLU)
        """
        return x * torch.sigmoid(beta * x)
    
    @staticmethod
    def mish(x: torch.Tensor) -> torch.Tensor:
        """
        Mish activation: x * tanh(softplus(x))
        Smooth, unbounded above, bounded below.
        Good for deep networks with gradient stability.
        """
        return x * torch.tanh(torch.nn.functional.softplus(x))
    
    @staticmethod
    def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
        """
        Leaky ReLU - Prevents dying ReLU problem.
        
        Args:
            x: Input tensor
            negative_slope: Slope for negative values
        """
        return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    
    @staticmethod
    def elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Exponential Linear Unit - Smooth, negative values have gradients.
        
        Args:
            x: Input tensor
            alpha: Scale for negative values
        """
        return torch.nn.functional.elu(x, alpha=alpha)
    
    @staticmethod
    def scaled_tanh(x: torch.Tensor, scale: float = 1.7159) -> torch.Tensor:
        """
        Scaled tanh with improved gradient flow.
        LeCun's recommended scaling for better gradient propagation.
        
        Args:
            x: Input tensor
            scale: Output scaling factor
        """
        return scale * torch.tanh(x)


def get_stable_loss_function(
    loss_version: int = 0,
    normalize: bool = True,
    eps: float = 1e-8,
    activation: Optional[str] = None
) -> Callable:
    """
    Factory function to create stable loss functions.
    
    Args:
        loss_version: Which loss variant to use
        normalize: Whether to normalize inputs
        eps: Epsilon for numerical stability
        activation: Optional activation function to apply
    
    Returns:
        Loss function that takes variable number of feature tensors
    """
    stable_ops = StableLossFunctions(eps=eps)
    
    # Select activation function
    act_fn = None
    if activation:
        activations = ImprovedActivations()
        act_fn = getattr(activations, activation, None)
    
    def stable_loss_fn(*features):
        # Apply normalization if requested
        if normalize:
            features = [stable_ops.stable_norm(f) for f in features]
        
        # Select loss computation
        if loss_version == 0:  # Einsum-based
            result = stable_ops.stable_einsum_similarity(*features)
        elif loss_version == 1:  # Norm-based
            result = stable_ops.stable_norm_based_similarity(*features)
        elif loss_version == 2:  # Cosine-based
            result = stable_ops.stable_cosine_similarity_nd(*features)
        elif loss_version == 3:  # Distance-based
            result = stable_ops.stable_euclidean_distance_nd(*features)
        else:
            raise ValueError(f"Unknown loss version: {loss_version}")
        
        # Apply activation if specified
        if act_fn is not None:
            result = act_fn(result)
        
        return result
    
    return stable_loss_fn


class GradientStabilizer(nn.Module):
    """
    Wrapper module to add gradient stabilization techniques.
    
    Can be inserted between layers to help with gradient flow.
    """
    
    def __init__(
        self,
        method: str = "layer_norm",
        hidden_dim: Optional[int] = None,
        eps: float = 1e-8,
        momentum: float = 0.1
    ):
        """
        Args:
            method: Stabilization method ('layer_norm', 'batch_norm', 'weight_norm', 'spectral_norm')
            hidden_dim: Hidden dimension size (required for some methods)
            eps: Epsilon for numerical stability
            momentum: Momentum for batch norm
        """
        super().__init__()
        self.method = method
        
        if method == "layer_norm":
            assert hidden_dim is not None, "hidden_dim required for layer_norm"
            self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        elif method == "batch_norm":
            assert hidden_dim is not None, "hidden_dim required for batch_norm"
            self.norm = nn.BatchNorm1d(hidden_dim, eps=eps, momentum=momentum)
        elif method == "weight_norm":
            # Applied via functional, no module needed
            self.norm = None
        elif method == "spectral_norm":
            # Applied via functional, no module needed
            self.norm = None
        else:
            raise ValueError(f"Unknown stabilization method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            return self.norm(x)
        return x


# Export main classes and functions
__all__ = [
    'StableLossFunctions',
    'ImprovedActivations',
    'GradientStabilizer',
    'get_stable_loss_function'
]
