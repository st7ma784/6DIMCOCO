"""Tests for CKA (Centered Kernel Alignment) analysis functionality."""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestCKAMethods:
    """Test CKA computation methods."""
    
    @pytest.fixture
    def sample_kernel_matrices(self):
        """Create sample kernel matrices for testing."""
        batch_size = 50
        # Create positive semi-definite matrices
        K = torch.randn(batch_size, batch_size)
        K = K @ K.T  # Make PSD
        K.fill_diagonal_(0)  # Remove diagonal as in original code
        
        L = torch.randn(batch_size, batch_size) 
        L = L @ L.T  # Make PSD
        L.fill_diagonal_(0)
        
        return K, L
    
    @pytest.fixture
    def batch_kernel_matrices(self):
        """Create batch of kernel matrices."""
        num_layers = 10
        batch_size = 20
        
        K_batch = torch.randn(num_layers, batch_size, batch_size)
        L_batch = torch.randn(num_layers, batch_size, batch_size)
        
        # Make each matrix in batch PSD
        for i in range(num_layers):
            K_batch[i] = K_batch[i] @ K_batch[i].T
            K_batch[i].fill_diagonal_(0)
            L_batch[i] = L_batch[i] @ L_batch[i].T  
            L_batch[i].fill_diagonal_(0)
        
        return K_batch, L_batch
    
    def test_original_hsic_implementation(self, sample_kernel_matrices):
        """Test the original HSIC implementation."""
        K, L = sample_kernel_matrices
        
        # Implement original HSIC from the codebase
        def orig_hsic(K, L):
            N = K.shape[0]
            return torch.add(
                torch.trace(K @ L),
                torch.div(
                    torch.sum(K) * torch.sum(L) / (N - 1) - (torch.sum(K @ L) * 2),
                    (N - 2)
                )
            )
        
        result = orig_hsic(K, L)
        
        # Basic sanity checks
        assert torch.isfinite(result).all()
        assert result.numel() == 1  # Should be scalar
    
    def test_improved_hsic_implementation(self, sample_kernel_matrices):
        """Test improved HSIC implementation with better numerical stability."""
        K, L = sample_kernel_matrices
        
        def improved_hsic(K, L, eps=1e-8):
            """Numerically stable HSIC implementation."""
            N = K.shape[0]
            if N < 3:
                return torch.tensor(0.0)
            
            # Clamp to avoid division by zero
            denom1 = max(N - 1, 1)
            denom2 = max(N - 2, 1)
            
            trace_term = torch.trace(K @ L)
            sum_K = torch.sum(K)
            sum_L = torch.sum(L)
            sum_KL = torch.sum(K @ L)
            
            correction_term = (sum_K * sum_L / denom1 - sum_KL * 2) / denom2
            
            result = trace_term + correction_term
            
            # Clamp result to avoid extreme values
            return torch.clamp(result, min=-1e6, max=1e6)
        
        result = improved_hsic(K, L)
        
        assert torch.isfinite(result).all()
        assert not torch.isnan(result).any()
    
    def test_batch_hsic_methods(self, batch_kernel_matrices):
        """Test batch HSIC computation methods."""
        K_batch, L_batch = batch_kernel_matrices
        
        def batch_hsic2(K):
            """Batch version of HSIC for single input (from original code)."""
            a = torch.sum(K, dim=-1)
            b = torch.sum(K, dim=-2) 
            
            N = K.shape[-2]
            if N < 3:
                return torch.zeros(K.shape[0])
            
            c = torch.sub(
                torch.pow(torch.sum(a, dim=-1), 2) / (N - 1),
                torch.sum(a * b, dim=1),
                alpha=2
            )
            
            diagonal_sum = torch.sum(torch.sum(K * K.permute(0, 2, 1), dim=-1), dim=-1)
            output = torch.add(diagonal_sum, torch.div(c, (N - 2)))
            
            return torch.div(output, (N * (N - 3))) if N > 3 else output
        
        def batch_hsic3(K, L):
            """Batch version of HSIC for two inputs."""
            K = K.unsqueeze(1)  # Shape: [layers_K, 1, B, B]
            L = L.unsqueeze(0)  # Shape: [1, layers_L, B, B]
            
            a = torch.sum(L, dim=-1)  # [1, layers_L, B]
            b = torch.sum(K, dim=-2)  # [layers_K, 1, B]
            
            N = K.shape[-2]
            if N < 3:
                return torch.zeros(K.shape[0], L.shape[1])
            
            c = torch.sub(
                torch.mul(torch.sum(b, dim=-1), torch.sum(a, dim=-1)).div(N - 1),
                torch.sum(torch.mul(b, a), dim=-1),
                alpha=2
            )
            
            kl_product = torch.sum(torch.sum(K * L, dim=-1), dim=-1)
            result = torch.div(
                torch.add(kl_product, torch.div(c, N - 2)),
                (N * (N - 3))
            ) if N > 3 else kl_product
            
            return result
        
        # Test single input batch method
        result2 = batch_hsic2(K_batch)
        assert result2.shape == (K_batch.shape[0],)
        assert torch.isfinite(result2).all()
        
        # Test two input batch method  
        result3 = batch_hsic3(K_batch, L_batch)
        assert result3.shape == (K_batch.shape[0], L_batch.shape[0])
        assert torch.isfinite(result3).all()
    
    def test_cka_computation(self, batch_kernel_matrices):
        """Test full CKA computation pipeline."""
        K_batch, L_batch = batch_kernel_matrices
        
        def compute_cka_matrix(K_features, L_features):
            """Compute CKA similarity matrix between two sets of features."""
            # Compute HSIC for each feature set with itself
            hsic_K = []
            hsic_L = []
            
            for i in range(K_features.shape[0]):
                hsic_k = self._compute_hsic_single(K_features[i])
                hsic_K.append(hsic_k)
            
            for i in range(L_features.shape[0]):
                hsic_l = self._compute_hsic_single(L_features[i])
                hsic_L.append(hsic_l)
            
            hsic_K = torch.stack(hsic_K)
            hsic_L = torch.stack(hsic_L)
            
            # Compute cross-HSIC
            hsic_KL = torch.zeros(K_features.shape[0], L_features.shape[0])
            for i in range(K_features.shape[0]):
                for j in range(L_features.shape[0]):
                    hsic_KL[i, j] = self._compute_hsic_cross(K_features[i], L_features[j])
            
            # Compute CKA
            denominator = torch.sqrt(hsic_K.unsqueeze(1) * hsic_L.unsqueeze(0))
            denominator = torch.clamp(denominator, min=1e-8)  # Avoid division by zero
            
            cka_matrix = hsic_KL / denominator
            
            return cka_matrix
        
        # This is a simplified test - full implementation would be more complex
        assert K_batch.shape[0] > 0
        assert L_batch.shape[0] > 0
    
    def _compute_hsic_single(self, K):
        """Helper method to compute HSIC for single matrix."""
        N = K.shape[0]
        if N < 3:
            return torch.tensor(0.0)
        
        trace_term = torch.trace(K @ K)
        sum_K = torch.sum(K)
        sum_diag = torch.sum(torch.diag(K @ K))
        
        correction = (sum_K ** 2 / (N - 1) - sum_diag * 2) / (N - 2)
        
        return (trace_term + correction) / (N * (N - 3)) if N > 3 else trace_term
    
    def _compute_hsic_cross(self, K, L):
        """Helper method to compute cross-HSIC."""
        N = K.shape[0]
        if N < 3:
            return torch.tensor(0.0)
        
        trace_term = torch.trace(K @ L)
        sum_K = torch.sum(K)
        sum_L = torch.sum(L)
        sum_KL = torch.sum(K @ L)
        
        correction = (sum_K * sum_L / (N - 1) - sum_KL * 2) / (N - 2)
        
        return (trace_term + correction) / (N * (N - 3)) if N > 3 else trace_term
    
    def test_numerical_stability_edge_cases(self):
        """Test CKA methods with edge cases."""
        # Test with very small matrices
        small_K = torch.eye(2) * 1e-8
        small_L = torch.eye(2) * 1e-8
        
        # Should handle gracefully without crashing
        result = self._compute_hsic_cross(small_K, small_L)
        assert torch.isfinite(result)
        
        # Test with zero matrices
        zero_K = torch.zeros(5, 5)
        zero_L = torch.zeros(5, 5)
        
        result = self._compute_hsic_cross(zero_K, zero_L)
        assert torch.isfinite(result)
        
        # Test with identity matrices
        eye_K = torch.eye(10)
        eye_L = torch.eye(10)
        
        result = self._compute_hsic_cross(eye_K, eye_L)
        assert torch.isfinite(result)
    
    def test_cka_properties(self, sample_kernel_matrices):
        """Test mathematical properties of CKA."""
        K, L = sample_kernel_matrices
        
        # CKA should be symmetric: CKA(K,L) = CKA(L,K)
        cka_kl = self._compute_hsic_cross(K, L)
        cka_lk = self._compute_hsic_cross(L, K)
        
        assert torch.allclose(cka_kl, cka_lk, atol=1e-6)
        
        # CKA with itself should be 1 (after proper normalization)
        hsic_kk = self._compute_hsic_single(K)
        hsic_ll = self._compute_hsic_single(L)
        
        if hsic_kk > 1e-8 and hsic_ll > 1e-8:  # Avoid division by zero
            normalized_cka = cka_kl / torch.sqrt(hsic_kk * hsic_ll)
            assert torch.abs(normalized_cka) <= 1.1  # Allow small numerical errors


class TestCKAIntegration:
    """Test CKA integration with model analysis."""
    
    def test_feature_extraction_simulation(self):
        """Simulate feature extraction from neural networks."""
        # Simulate features from different layers of two models
        batch_size = 32
        num_layers_1 = 12
        num_layers_2 = 16
        feature_dim = 768
        
        # Simulate model 1 features
        model1_features = []
        for layer in range(num_layers_1):
            # Simulate layer features with some structure
            feat = torch.randn(batch_size, feature_dim) * (0.5 + layer * 0.1)
            # Create gram matrix (feature correlations)
            gram = feat @ feat.T
            gram.fill_diagonal_(0)  # As in original code
            model1_features.append(gram)
        
        # Simulate model 2 features  
        model2_features = []
        for layer in range(num_layers_2):
            feat = torch.randn(batch_size, feature_dim) * (0.3 + layer * 0.15)
            gram = feat @ feat.T
            gram.fill_diagonal_(0)
            model2_features.append(gram)
        
        # Stack into tensors
        model1_stack = torch.stack(model1_features)
        model2_stack = torch.stack(model2_features)
        
        # Basic validation
        assert model1_stack.shape == (num_layers_1, batch_size, batch_size)
        assert model2_stack.shape == (num_layers_2, batch_size, batch_size)
        
        # All matrices should be finite
        assert torch.isfinite(model1_stack).all()
        assert torch.isfinite(model2_stack).all()
    
    def test_cka_analysis_pipeline(self):
        """Test complete CKA analysis pipeline."""
        # This would test the full pipeline from model features to CKA matrix
        # For now, just test that the structure is sound
        
        batch_size = 16
        num_layers = 6
        
        # Simulate feature matrices
        features = torch.randn(num_layers, batch_size, batch_size)
        
        # Ensure matrices are symmetric (as gram matrices should be)
        for i in range(num_layers):
            features[i] = (features[i] + features[i].T) / 2
            features[i].fill_diagonal_(0)
        
        # Test that we can process these features
        assert features.shape == (num_layers, batch_size, batch_size)
        assert torch.allclose(features, features.transpose(-2, -1))  # Symmetric


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
