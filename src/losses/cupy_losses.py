"""CuPy-accelerated loss functions for high-performance GPU computation."""

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

import torch
import numpy as np
from functools import reduce
from typing import Tuple, Optional
from .base_loss import BaseLoss, LossRegistry


def torch_to_cupy(tensor: torch.Tensor) -> 'cp.ndarray':
    """Convert PyTorch tensor to CuPy array."""
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy not available")
    
    if tensor.is_cuda:
        return cp.asarray(tensor.detach())
    else:
        return cp.asarray(tensor.detach().cpu().numpy())


def cupy_to_torch(array: 'cp.ndarray', device: torch.device = None) -> torch.Tensor:
    """Convert CuPy array to PyTorch tensor."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        # Direct GPU memory transfer
        return torch.as_tensor(array, device=device)
    else:
        return torch.from_numpy(cp.asnumpy(array)).to(device)


@LossRegistry.register("cupy_einsum")
class CuPyEinsumLoss(BaseLoss):
    """CuPy-accelerated Einstein summation loss."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for CuPyEinsumLoss")
    
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """Compute einsum loss using CuPy for GPU acceleration."""
        if len(features) != 6:
            raise ValueError("CuPyEinsumLoss requires exactly 6 feature tensors")
        
        # Convert to CuPy arrays
        cp_features = [torch_to_cupy(feat) for feat in features]
        I, C1, C2, C3, C4, C5 = cp_features
        
        # Perform einsum operations on GPU
        left_term = cp.einsum("az,bz,cz->abcz", I, C1, C2)
        right_term = cp.einsum("az,bz,cz->abcz", C3, C4, C5)
        result = cp.einsum("abcz,defz->abcdef", left_term, right_term)
        
        # Convert back to PyTorch
        return cupy_to_torch(result, device=features[0].device)


@LossRegistry.register("cupy_norm_based")
class CuPyNormBasedLoss(BaseLoss):
    """CuPy-accelerated norm-based loss with JSE shrinkage."""
    
    def __init__(self, 
                 variant: str = "v4",
                 jse_shrinkage: bool = False,
                 shrinkage_factor: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy is required for CuPyNormBasedLoss")
        
        self.variant = variant
        self.jse_shrinkage = jse_shrinkage
        self.shrinkage_factor = shrinkage_factor
    
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """Compute norm-based loss with optional JSE shrinkage."""
        if len(features) != 6:
            raise ValueError("CuPyNormBasedLoss requires exactly 6 feature tensors")
        
        # Convert to CuPy for GPU computation
        cp_features = [torch_to_cupy(feat) for feat in features]
        
        if self.normalize:
            cp_features = self._normalize_features_cupy(*cp_features)
        
        # Compute mean using CuPy's optimized operations
        mean = self._compute_mean_cupy(*cp_features)
        
        # Apply JSE shrinkage if enabled
        if self.jse_shrinkage:
            mean = self._apply_jse_shrinkage_cupy(mean, *cp_features)
        
        # Compute similarities
        result = self._compute_similarities_cupy(mean, *cp_features)
        
        return cupy_to_torch(result, device=features[0].device)
    
    def _normalize_features_cupy(self, *features) -> Tuple:
        """Normalize features using CuPy."""
        normalized = []
        for feat in features:
            norm = cp.linalg.norm(feat, axis=-1, keepdims=True)
            norm = cp.maximum(norm, self.eps)  # Clamp for stability
            normalized.append(feat / norm)
        return tuple(normalized)
    
    def _compute_mean_cupy(self, *features) -> 'cp.ndarray':
        """Compute mean of features using optimized CuPy operations."""
        # Create expanded views for 6D computation
        expanded_features = []
        for i, feat in enumerate(features):
            shape = [1] * 6
            shape[i] = feat.shape[0]
            shape.append(feat.shape[-1])
            expanded_feat = feat.reshape(*shape)
            expanded_features.append(expanded_feat)
        
        # Sum all features - CuPy handles memory efficiently
        mean = cp.sum(cp.stack(expanded_features), axis=0)
        return mean
    
    def _apply_jse_shrinkage_cupy(self, mean: 'cp.ndarray', *features) -> 'cp.ndarray':
        """Apply James-Stein Estimator shrinkage to the mean."""
        # Compute empirical variance
        variance = cp.var(cp.stack([feat.flatten() for feat in features]))
        
        # JSE shrinkage: shrink towards zero
        shrinkage = self.shrinkage_factor * variance / (variance + cp.mean(mean**2))
        shrunk_mean = (1 - shrinkage) * mean
        
        return shrunk_mean
    
    def _compute_similarities_cupy(self, mean: 'cp.ndarray', *features) -> 'cp.ndarray':
        """Compute similarities using CuPy's optimized operations."""
        # Normalize mean
        mean_norm = cp.linalg.norm(mean, axis=-1, keepdims=True)
        mean_norm = cp.maximum(mean_norm, self.eps)
        mean_normalized = mean / mean_norm
        
        # Compute similarities with each feature
        similarities = []
        for i, feat in enumerate(features):
            # Create appropriate view
            shape = [1] * 6
            shape[i] = feat.shape[0]
            shape.append(feat.shape[-1])
            feat_expanded = feat.reshape(*shape)
            
            # Compute dot product similarity
            similarity = cp.sum(mean_normalized * feat_expanded, axis=-1)
            similarities.append(similarity)
        
        # Sum all similarities
        result = cp.sum(cp.stack(similarities), axis=0)
        return result


class CuPyPerformanceProfiler:
    """Profile CuPy vs PyTorch performance for loss functions."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_loss_function(self, 
                               loss_name: str,
                               batch_sizes: list = [8, 16, 32, 64],
                               feature_dim: int = 512,
                               num_runs: int = 10):
        """Benchmark CuPy vs PyTorch implementations."""
        if not CUPY_AVAILABLE:
            print("CuPy not available for benchmarking")
            return
        
        results = {
            'pytorch': {},
            'cupy': {},
            'speedup': {}
        }
        
        for batch_size in batch_sizes:
            print(f"Benchmarking batch_size={batch_size}")
            
            # Create test features
            features = [torch.randn(batch_size, feature_dim).cuda() 
                       for _ in range(6)]
            
            # Benchmark PyTorch version
            pytorch_loss = self._get_pytorch_loss(loss_name)
            pytorch_time = self._time_function(pytorch_loss, features, num_runs)
            
            # Benchmark CuPy version
            cupy_loss = self._get_cupy_loss(loss_name)
            cupy_time = self._time_function(cupy_loss, features, num_runs)
            
            results['pytorch'][batch_size] = pytorch_time
            results['cupy'][batch_size] = cupy_time
            results['speedup'][batch_size] = pytorch_time / cupy_time
            
            print(f"  PyTorch: {pytorch_time:.4f}s")
            print(f"  CuPy: {cupy_time:.4f}s") 
            print(f"  Speedup: {results['speedup'][batch_size]:.2f}x")
        
        self.results[loss_name] = results
        return results
    
    def _get_pytorch_loss(self, loss_name: str):
        """Get PyTorch version of loss function."""
        from .ndim_losses import NormBasedLoss, EinsumLoss
        
        if loss_name == 'norm_based':
            return NormBasedLoss(variant='v4', normalize=True)
        elif loss_name == 'einsum':
            return EinsumLoss(normalize=True)
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
    
    def _get_cupy_loss(self, loss_name: str):
        """Get CuPy version of loss function."""
        if loss_name == 'norm_based':
            return CuPyNormBasedLoss(variant='v4', normalize=True)
        elif loss_name == 'einsum':
            return CuPyEinsumLoss(normalize=True)
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
    
    def _time_function(self, loss_fn, features, num_runs: int) -> float:
        """Time a loss function over multiple runs."""
        import time
        
        # Warmup
        for _ in range(3):
            _ = loss_fn(*features)
        
        # Synchronize GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Time actual runs
        start_time = time.time()
        for _ in range(num_runs):
            _ = loss_fn(*features)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        return (end_time - start_time) / num_runs


def get_memory_usage_comparison():
    """Compare memory usage between PyTorch and CuPy implementations."""
    if not CUPY_AVAILABLE:
        return "CuPy not available"
    
    batch_size = 32
    feature_dim = 512
    
    # Create test data
    features = [torch.randn(batch_size, feature_dim).cuda() for _ in range(6)]
    
    # Measure PyTorch memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    pytorch_loss = NormBasedLoss(variant='v4', normalize=True)
    _ = pytorch_loss(*features)
    
    pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # Measure CuPy memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    cupy_loss = CuPyNormBasedLoss(variant='v4', normalize=True)
    _ = cupy_loss(*features)
    
    cupy_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    return {
        'pytorch_memory_mb': pytorch_memory,
        'cupy_memory_mb': cupy_memory,
        'memory_ratio': pytorch_memory / cupy_memory
    }
