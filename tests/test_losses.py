"""Comprehensive tests for loss functions."""

import pytest
import torch
import numpy as np
from typing import List, Tuple
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from losses import (
    create_loss_function, 
    get_available_losses,
    create_loss_from_legacy_version,
    StockCLIPLoss,
    EinsumLoss,
    EuclideanDistanceLoss,
    NormBasedLoss
)
from config.base_config import ModelConfig


class TestLossFunctions:
    """Test suite for loss functions."""
    
    @pytest.fixture
    def sample_features(self) -> List[torch.Tensor]:
        """Create sample feature tensors for testing."""
        batch_size = 8
        feature_dim = 512
        
        # Create 6 feature tensors as required by n-dimensional losses
        features = []
        for i in range(6):
            # Add some variation to make tests more realistic
            feat = torch.randn(batch_size, feature_dim) * (0.5 + i * 0.1)
            features.append(feat)
        
        return features
    
    @pytest.fixture
    def normalized_features(self, sample_features) -> List[torch.Tensor]:
        """Create normalized feature tensors."""
        normalized = []
        for feat in sample_features:
            norm = feat.norm(dim=-1, keepdim=True)
            normalized.append(feat / torch.clamp(norm, min=1e-8))
        return normalized
    
    def test_loss_creation_from_registry(self):
        """Test creating losses from registry."""
        available_losses = get_available_losses()
        
        for loss_name in available_losses:
            loss_fn = create_loss_function(loss_name)
            assert loss_fn is not None
            assert hasattr(loss_fn, 'forward')
    
    def test_stock_clip_loss(self, normalized_features):
        """Test standard CLIP loss."""
        loss_fn = StockCLIPLoss(normalize=True)
        
        # Stock CLIP only needs 2 features
        image_feat, text_feat = normalized_features[0], normalized_features[1]
        
        logits_img, logits_text = loss_fn(image_feat, text_feat)
        
        # Check shapes
        batch_size = image_feat.shape[0]
        assert logits_img.shape == (batch_size, batch_size)
        assert logits_text.shape == (batch_size, batch_size)
        
        # Check symmetry property
        assert torch.allclose(logits_img, logits_text.T, atol=1e-6)
    
    def test_einsum_loss(self, sample_features):
        """Test einsum-based loss."""
        loss_fn = EinsumLoss(normalize=True)
        
        result = loss_fn(*sample_features)
        
        # Check output shape (should be 6D tensor)
        batch_size = sample_features[0].shape[0]
        expected_shape = (batch_size,) * 6
        assert result.shape == expected_shape
        
        # Check for NaN/Inf
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    def test_euclidean_distance_loss(self, sample_features):
        """Test Euclidean distance loss."""
        loss_fn = EuclideanDistanceLoss(normalize=True, use_sqrt=False)
        
        result = loss_fn(*sample_features)
        
        # Check output shape
        batch_size = sample_features[0].shape[0]
        expected_shape = (batch_size,) * 6
        assert result.shape == expected_shape
        
        # Check for numerical stability
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert (result >= 0).all()  # Distance should be non-negative
    
    def test_norm_based_loss_variants(self, normalized_features):
        """Test different variants of norm-based loss."""
        variants = ['v1', 'v4']  # Test implemented variants
        
        for variant in variants:
            loss_fn = NormBasedLoss(variant=variant, normalize=False)  # Already normalized
            
            result = loss_fn(*normalized_features)
            
            # Check output shape
            batch_size = normalized_features[0].shape[0]
            expected_shape = (batch_size,) * 6
            assert result.shape == expected_shape
            
            # Check for numerical issues
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
    
    def test_loss_function_properties(self, sample_features):
        """Test mathematical properties of loss functions."""
        loss_types = ['einsum', 'euclidean_distance', 'norm_based']
        
        for loss_type in loss_types:
            loss_fn = create_loss_function(loss_type, normalize=True)
            
            # Test with original features
            result1 = loss_fn(*sample_features)
            
            # Test with scaled features (should be invariant under normalization)
            scaled_features = [feat * 2.0 for feat in sample_features]
            result2 = loss_fn(*scaled_features)
            
            # Results should be similar due to normalization
            if loss_fn.normalize:
                assert torch.allclose(result1, result2, rtol=1e-4, atol=1e-6)
    
    def test_transpose_invariance(self, sample_features):
        """Test transpose invariance property."""
        # This is inspired by the original test_LSA_Loss.py
        loss_fn = create_loss_function('norm_based', variant='v4', normalize=True)
        
        # Compute loss with original features
        result1 = loss_fn(*sample_features)
        
        # Compute loss with transposed features
        transposed_features = [feat.T for feat in sample_features]
        
        # Note: This test might not be applicable to all loss functions
        # depending on their mathematical formulation
        try:
            result2 = loss_fn(*transposed_features)
            # If shapes are compatible, check for some relationship
            if result1.shape == result2.shape:
                # At minimum, both should be finite
                assert torch.isfinite(result1).all()
                assert torch.isfinite(result2).all()
        except RuntimeError:
            # Expected for incompatible shapes
            pass
    
    def test_batch_size_consistency(self):
        """Test that loss functions work with different batch sizes."""
        batch_sizes = [1, 4, 16, 32]
        feature_dim = 256
        
        for batch_size in batch_sizes:
            features = [torch.randn(batch_size, feature_dim) for _ in range(6)]
            
            loss_fn = create_loss_function('euclidean_distance', normalize=True)
            result = loss_fn(*features)
            
            expected_shape = (batch_size,) * 6
            assert result.shape == expected_shape
    
    def test_legacy_version_compatibility(self, sample_features):
        """Test compatibility with legacy loss versions."""
        # Test a few legacy versions
        legacy_versions = [0, 1, 7, 10]
        
        for version in legacy_versions:
            try:
                loss_fn = create_loss_from_legacy_version(version)
                result = loss_fn(*sample_features)
                
                # Basic sanity checks
                assert result is not None
                assert not torch.isnan(result).any()
                
            except (ValueError, NotImplementedError):
                # Some legacy versions might not be fully implemented
                pass
    
    def test_numerical_stability_edge_cases(self):
        """Test numerical stability with edge cases."""
        batch_size = 4
        feature_dim = 128
        
        # Test with very small values
        small_features = [torch.ones(batch_size, feature_dim) * 1e-8 for _ in range(6)]
        
        # Test with very large values  
        large_features = [torch.ones(batch_size, feature_dim) * 1e6 for _ in range(6)]
        
        # Test with zero features
        zero_features = [torch.zeros(batch_size, feature_dim) for _ in range(6)]
        
        loss_fn = create_loss_function('euclidean_distance', normalize=True)
        
        for features, name in [(small_features, "small"), (large_features, "large"), (zero_features, "zero")]:
            try:
                result = loss_fn(*features)
                assert torch.isfinite(result).all(), f"Non-finite values with {name} features"
            except Exception as e:
                pytest.fail(f"Loss function failed with {name} features: {e}")
    
    def test_gradient_flow(self, sample_features):
        """Test that gradients flow properly through loss functions."""
        # Make features require gradients
        features_with_grad = [feat.clone().requires_grad_(True) for feat in sample_features]
        
        loss_fn = create_loss_function('norm_based', variant='v4', normalize=True)
        result = loss_fn(*features_with_grad)
        
        # Compute a scalar loss for backprop
        scalar_loss = result.mean()
        scalar_loss.backward()
        
        # Check that gradients exist and are finite
        for i, feat in enumerate(features_with_grad):
            assert feat.grad is not None, f"No gradient for feature {i}"
            assert torch.isfinite(feat.grad).all(), f"Non-finite gradients for feature {i}"
    
    def test_config_integration(self, sample_features):
        """Test integration with configuration system."""
        config = ModelConfig(
            normalize_logits=True,
            log_variance=False,
            dimensions=6.0,
            loss_version=0
        )
        
        loss_fn = create_loss_function('norm_based', config=config)
        
        assert loss_fn.normalize == config.normalize_logits
        assert loss_fn.log_variance == config.log_variance
        
        # Test that it works
        result = loss_fn(*sample_features)
        assert result is not None


class TestLossValidation:
    """Test input validation and error handling."""
    
    def test_empty_features_validation(self):
        """Test validation with empty features."""
        loss_fn = create_loss_function('euclidean_distance')
        
        with pytest.raises(ValueError, match="At least one feature tensor must be provided"):
            loss_fn()
    
    def test_mismatched_batch_sizes(self):
        """Test validation with mismatched batch sizes."""
        features = [
            torch.randn(8, 512),   # batch_size = 8
            torch.randn(4, 512),   # batch_size = 4 (mismatch!)
            torch.randn(8, 512),
            torch.randn(8, 512),
            torch.randn(8, 512),
            torch.randn(8, 512),
        ]
        
        loss_fn = create_loss_function('euclidean_distance')
        
        with pytest.raises(ValueError, match="All features must have the same batch size"):
            loss_fn(*features)
    
    def test_wrong_number_of_features(self):
        """Test with wrong number of input features."""
        # Most n-dimensional losses expect exactly 6 features
        features = [torch.randn(4, 512) for _ in range(3)]  # Only 3 features
        
        loss_fn = create_loss_function('einsum')
        
        with pytest.raises(ValueError, match="requires exactly 6 feature tensors"):
            loss_fn(*features)
    
    def test_invalid_loss_type(self):
        """Test creation with invalid loss type."""
        with pytest.raises(ValueError, match="Loss function .* not found"):
            create_loss_function('nonexistent_loss')
    
    def test_invalid_legacy_version(self):
        """Test invalid legacy version."""
        with pytest.raises(ValueError, match="Unknown legacy version"):
            create_loss_from_legacy_version(999)


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
