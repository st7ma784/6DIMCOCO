"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def device():
    """Get the best available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def seed():
    """Set random seed for reproducible tests."""
    seed_value = 42
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
    return seed_value


@pytest.fixture
def small_batch_features():
    """Create small batch of features for quick testing."""
    batch_size = 4
    feature_dim = 64
    num_features = 6
    
    features = []
    for i in range(num_features):
        feat = torch.randn(batch_size, feature_dim)
        features.append(feat)
    
    return features


@pytest.fixture
def medium_batch_features():
    """Create medium batch of features for thorough testing."""
    batch_size = 16
    feature_dim = 256
    num_features = 6
    
    features = []
    for i in range(num_features):
        feat = torch.randn(batch_size, feature_dim)
        features.append(feat)
    
    return features


@pytest.fixture
def large_batch_features():
    """Create large batch of features for performance testing."""
    batch_size = 64
    feature_dim = 512
    num_features = 6
    
    features = []
    for i in range(num_features):
        feat = torch.randn(batch_size, feature_dim)
        features.append(feat)
    
    return features


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark GPU tests."""
    for item in items:
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
