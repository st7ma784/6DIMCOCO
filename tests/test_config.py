"""Tests for configuration management system."""

import pytest
import torch
import tempfile
import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.base_config import (
    ModelConfig,
    TrainingConfig, 
    DataConfig,
    ExperimentConfig,
    get_device_config
)


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_default_initialization(self):
        """Test default config initialization."""
        config = ModelConfig()
        
        assert config.embed_dim == 512
        assert config.dimensions == 6.0
        assert config.normalize_logits is True
        assert config.transformer_layers == 4
    
    def test_custom_initialization(self):
        """Test custom config initialization."""
        config = ModelConfig(
            embed_dim=1024,
            dimensions=4.0,
            transformer_layers=8
        )
        
        assert config.embed_dim == 1024
        assert config.dimensions == 4.0
        assert config.transformer_layers == 8
    
    def test_invalid_dimensions(self):
        """Test validation of invalid dimensions."""
        with pytest.raises(ValueError, match="Unsupported dimensions"):
            ModelConfig(dimensions=7.0)
    
    def test_valid_dimensions(self):
        """Test all valid dimension values."""
        valid_dims = [3, 3.5, 4, 6, -1, 0]
        
        for dim in valid_dims:
            config = ModelConfig(dimensions=dim)
            assert config.dimensions == dim


class TestTrainingConfig:
    """Test TrainingConfig class."""
    
    def test_default_values(self):
        """Test default training configuration."""
        config = TrainingConfig()
        
        assert config.learning_rate == 2e-3
        assert config.train_batch_size == 64
        assert config.max_epochs == 20
        assert config.precision == 16
    
    def test_custom_values(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            learning_rate=1e-4,
            train_batch_size=32,
            precision=32
        )
        
        assert config.learning_rate == 1e-4
        assert config.train_batch_size == 32
        assert config.precision == 32


class TestDataConfig:
    """Test DataConfig class."""
    
    def test_default_values(self):
        """Test default data configuration."""
        config = DataConfig()
        
        assert config.cache_dir == "."
        assert config.num_workers == 4
        assert config.chinese_mode is False
        assert config.image_size == 224
    
    def test_chinese_mode(self):
        """Test Chinese language configuration."""
        config = DataConfig(
            chinese_mode=True,
            tokenizer_name="bert-base-chinese"
        )
        
        assert config.chinese_mode is True
        assert config.tokenizer_name == "bert-base-chinese"


class TestExperimentConfig:
    """Test ExperimentConfig class."""
    
    def test_default_initialization(self):
        """Test default experiment config."""
        config = ExperimentConfig()
        
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.data, DataConfig)
        assert config.seed == 42
    
    def test_nested_config_access(self):
        """Test accessing nested configuration values."""
        config = ExperimentConfig()
        
        # Test accessing nested values
        assert config.model.embed_dim == 512
        assert config.training.learning_rate == 2e-3
        assert config.data.num_workers == 4
    
    def test_to_dict_conversion(self):
        """Test converting config to dictionary."""
        config = ExperimentConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "training" in config_dict
        assert "data" in config_dict
        assert config_dict["seed"] == 42
    
    def test_from_dict_conversion(self):
        """Test creating config from dictionary."""
        original_config = ExperimentConfig()
        config_dict = original_config.to_dict()
        
        # Modify some values
        config_dict["model"]["embed_dim"] = 1024
        config_dict["training"]["learning_rate"] = 1e-4
        config_dict["seed"] = 123
        
        new_config = ExperimentConfig.from_dict(config_dict)
        
        assert new_config.model.embed_dim == 1024
        assert new_config.training.learning_rate == 1e-4
        assert new_config.seed == 123
    
    def test_roundtrip_conversion(self):
        """Test roundtrip dict conversion preserves values."""
        original_config = ExperimentConfig()
        config_dict = original_config.to_dict()
        restored_config = ExperimentConfig.from_dict(config_dict)
        
        # Check that key values are preserved
        assert original_config.model.embed_dim == restored_config.model.embed_dim
        assert original_config.training.learning_rate == restored_config.training.learning_rate
        assert original_config.data.num_workers == restored_config.data.num_workers
        assert original_config.seed == restored_config.seed


class TestDeviceConfig:
    """Test device configuration utilities."""
    
    def test_get_device_config(self):
        """Test device configuration detection."""
        device_config = get_device_config()
        
        assert "accelerator" in device_config
        assert "devices" in device_config
        assert "precision" in device_config
        
        # Should return valid values
        assert device_config["accelerator"] in ["auto", "cpu", "gpu"]
        assert isinstance(device_config["precision"], int)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_specific_config(self):
        """Test GPU-specific configuration."""
        device_config = get_device_config()
        
        # Should detect CUDA availability
        if torch.cuda.is_available():
            assert device_config["precision"] in [16, 32]


class TestConfigValidation:
    """Test configuration validation and error handling."""
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Test invalid transformer heads (should be positive)
        with pytest.raises((ValueError, AssertionError)):
            ModelConfig(transformer_heads=0)
        
        # Test invalid embed_dim (should be positive)
        with pytest.raises((ValueError, AssertionError)):
            ModelConfig(embed_dim=-1)
    
    def test_training_config_validation(self):
        """Test training configuration validation."""
        # Test invalid learning rate
        config = TrainingConfig(learning_rate=-1.0)
        # Note: We might want to add validation for this
        
        # Test invalid batch size
        config = TrainingConfig(train_batch_size=0)
        # Note: We might want to add validation for this
    
    def test_data_config_validation(self):
        """Test data configuration validation."""
        # Test invalid num_workers
        config = DataConfig(num_workers=-1)
        # Note: We might want to add validation for this
        
        # Test invalid image_size
        config = DataConfig(image_size=0)
        # Note: We might want to add validation for this


class TestConfigSerialization:
    """Test configuration serialization and deserialization."""
    
    def test_json_serialization(self):
        """Test JSON serialization of configs."""
        config = ExperimentConfig()
        config_dict = config.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(config_dict)
        restored_dict = json.loads(json_str)
        
        assert restored_dict == config_dict
    
    def test_file_save_load(self):
        """Test saving and loading config from file."""
        config = ExperimentConfig()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config.to_dict(), f)
            temp_path = f.name
        
        try:
            # Load from file
            with open(temp_path, 'r') as f:
                loaded_dict = json.load(f)
            
            restored_config = ExperimentConfig.from_dict(loaded_dict)
            
            # Verify key values are preserved
            assert config.model.embed_dim == restored_config.model.embed_dim
            assert config.training.learning_rate == restored_config.training.learning_rate
            
        finally:
            # Clean up
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
