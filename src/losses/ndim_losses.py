"""N-dimensional loss functions for CLIP training."""

import torch
import torch.nn.functional as F
from functools import reduce
from typing import Tuple
from .base_loss import BaseLoss, MultiDimensionalLoss, LossRegistry


@LossRegistry.register("stock_clip")
class StockCLIPLoss(BaseLoss):
    """Standard CLIP contrastive loss."""
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Compute standard CLIP loss."""
        image_features, text_features = self._normalize_features(image_features, text_features)
        
        # Compute similarity matrix
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T
        
        return logits_per_image, logits_per_text


@LossRegistry.register("einsum")
class EinsumLoss(MultiDimensionalLoss):
    """Einstein summation based loss for n-dimensional features."""
    
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Compute einsum-based loss.
        Original: torch.einsum("abcz,defz->abcdef", ...)
        """
        if len(features) != 6:
            raise ValueError("EinsumLoss requires exactly 6 feature tensors")
        
        I, C1, C2, C3, C4, C5 = features
        
        # Compute the einsum operation with proper batching
        left_term = torch.einsum("az,bz,cz->abcz", I, C1, C2)
        right_term = torch.einsum("az,bz,cz->abcz", C3, C4, C5)
        
        result = torch.einsum("abcz,defz->abcdef", left_term, right_term)
        
        return self._check_for_anomalies(result, "einsum_loss")


@LossRegistry.register("euclidean_distance")
class EuclideanDistanceLoss(MultiDimensionalLoss):
    """Euclidean distance based loss with numerical stability."""
    
    def __init__(self, use_sqrt: bool = True, **kwargs):
        """
        Initialize Euclidean distance loss.
        
        Args:
            use_sqrt: Whether to apply square root (can cause numerical issues)
        """
        super().__init__(**kwargs)
        self.use_sqrt = use_sqrt
    
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean distance based loss."""
        if len(features) != 6:
            raise ValueError("EuclideanDistanceLoss requires exactly 6 feature tensors")
        
        # Expand features for n-dimensional computation
        expanded_features = self._expand_for_dimensions(*features)
        
        # Compute sum of features
        feature_sum = reduce(torch.add, [
            feat.view(feat.shape[0], 1, 1, 1, 1, 1, -1) if i == 0 else
            feat.view(1, feat.shape[0] if i == 1 else 1, 
                     feat.shape[0] if i == 2 else 1,
                     feat.shape[0] if i == 3 else 1,
                     feat.shape[0] if i == 4 else 1,
                     feat.shape[0] if i == 5 else 1, -1)
            for i, feat in enumerate(features)
        ])
        
        # Compute sum of squares
        squares_sum = reduce(torch.add, [
            torch.pow(feat, 2).view(feat.shape[0], 1, 1, 1, 1, 1, -1) if i == 0 else
            torch.pow(feat, 2).view(1, feat.shape[0] if i == 1 else 1,
                                   feat.shape[0] if i == 2 else 1,
                                   feat.shape[0] if i == 3 else 1,
                                   feat.shape[0] if i == 4 else 1,
                                   feat.shape[0] if i == 5 else 1, -1)
            for i, feat in enumerate(features)
        ])
        
        # Compute distance: ||sum||^2 - sum(||x||^2)
        distance_squared = squares_sum - torch.pow(feature_sum, 2) / 6
        
        # Add epsilon for numerical stability
        distance_squared = torch.clamp(distance_squared, min=self.eps)
        
        if self.use_sqrt:
            result = torch.sqrt(distance_squared)
        else:
            result = distance_squared
        
        # Reduce over feature dimension
        result = torch.mean(result, dim=-1)
        
        return self._check_for_anomalies(result, "euclidean_distance_loss")


@LossRegistry.register("norm_based")
class NormBasedLoss(MultiDimensionalLoss):
    """Norm-based loss functions with various formulations."""
    
    def __init__(self, variant: str = "v4", **kwargs):
        """
        Initialize norm-based loss.
        
        Args:
            variant: Which variant to use (v1-v9)
        """
        super().__init__(**kwargs)
        self.variant = variant
        
        # Map variants to their implementations
        self.variant_map = {
            "v1": self._compute_v1,
            "v2": self._compute_v2,
            "v3": self._compute_v3,
            "v4": self._compute_v4,
            "v5": self._compute_v5,
        }
        
        if variant not in self.variant_map:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(self.variant_map.keys())}")
    
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """Compute norm-based loss using specified variant."""
        if len(features) != 6:
            raise ValueError("NormBasedLoss requires exactly 6 feature tensors")
        
        # Normalize features if required
        if self.normalize:
            features = self._normalize_features(*features)
        
        return self.variant_map[self.variant](*features)
    
    def _compute_v1(self, *features: torch.Tensor) -> torch.Tensor:
        """Compute norm-based loss variant 1."""
        # Compute mean of all features
        mean = reduce(torch.add, [
            feat.view(feat.shape[0], 1, 1, 1, 1, 1, -1) if i == 0 else
            feat.view(1, feat.shape[0] if i == 1 else 1,
                     feat.shape[0] if i == 2 else 1,
                     feat.shape[0] if i == 3 else 1,
                     feat.shape[0] if i == 4 else 1,
                     feat.shape[0] if i == 5 else 1, -1)
            for i, feat in enumerate(features)
        ])
        
        # Normalize mean
        mean_norm = mean.norm(dim=-1, keepdim=True)
        mean_norm = torch.clamp(mean_norm, min=self.eps)
        scaled_norm = mean / mean_norm
        
        # Compute dot product
        result = torch.sum(torch.mul(scaled_norm, mean), dim=-1)
        
        return self._check_for_anomalies(result, "norm_v1_loss")
    
    def _compute_v4(self, *features: torch.Tensor) -> torch.Tensor:
        """Compute norm-based loss variant 4 (most stable)."""
        # Compute mean of normalized features
        mean = reduce(torch.add, [
            feat.view(feat.shape[0], 1, 1, 1, 1, 1, -1) if i == 0 else
            feat.view(1, feat.shape[0] if i == 1 else 1,
                     feat.shape[0] if i == 2 else 1,
                     feat.shape[0] if i == 3 else 1,
                     feat.shape[0] if i == 4 else 1,
                     feat.shape[0] if i == 5 else 1, -1)
            for i, feat in enumerate(features)
        ])
        
        # Normalize mean
        mean_norm = mean.norm(dim=-1, keepdim=True)
        mean_norm = torch.clamp(mean_norm, min=self.eps)
        mean_normalized = mean / mean_norm
        
        # Compute similarities with each feature
        similarities = []
        for i, feat in enumerate(features):
            feat_expanded = feat.view(feat.shape[0], 1, 1, 1, 1, 1, -1) if i == 0 else \
                           feat.view(1, feat.shape[0] if i == 1 else 1,
                                   feat.shape[0] if i == 2 else 1,
                                   feat.shape[0] if i == 3 else 1,
                                   feat.shape[0] if i == 4 else 1,
                                   feat.shape[0] if i == 5 else 1, -1)
            
            similarity = torch.sum(torch.mul(mean_normalized, feat_expanded), dim=-1)
            similarities.append(similarity)
        
        # Sum all similarities
        result = reduce(torch.add, similarities)
        
        return self._check_for_anomalies(result, "norm_v4_loss")
    
    def _compute_v2(self, *features: torch.Tensor) -> torch.Tensor:
        """Compute norm-based loss variant 2."""
        # Implementation for variant 2
        pass
    
    def _compute_v3(self, *features: torch.Tensor) -> torch.Tensor:
        """Compute norm-based loss variant 3."""
        # Implementation for variant 3
        pass
    
    def _compute_v5(self, *features: torch.Tensor) -> torch.Tensor:
        """Compute norm-based loss variant 5."""
        # Implementation for variant 5
        pass


@LossRegistry.register("cosine_similarity")
class CosineSimilarityLoss(MultiDimensionalLoss):
    """Cosine similarity based loss for multi-dimensional features."""
    
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity based loss."""
        if len(features) != 6:
            raise ValueError("CosineSimilarityLoss requires exactly 6 feature tensors")
        
        # Normalize all features
        normalized_features = self._normalize_features(*features)
        
        # Compute pairwise cosine similarities
        total_similarity = torch.zeros_like(normalized_features[0][:, 0])
        
        for i, feat_i in enumerate(normalized_features):
            for j, feat_j in enumerate(normalized_features):
                if i != j:  # Don't compute self-similarity
                    similarity = F.cosine_similarity(feat_i, feat_j, dim=-1)
                    total_similarity = total_similarity + similarity
        
        # Average over all pairs
        num_pairs = len(features) * (len(features) - 1)
        avg_similarity = total_similarity / num_pairs
        
        return self._check_for_anomalies(avg_similarity, "cosine_similarity_loss")
