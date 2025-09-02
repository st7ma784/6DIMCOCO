"""Base configuration classes for 6DIMCOCO framework."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for CLIP model architecture."""
    
    # Model dimensions
    embed_dim: int = 512
    context_length: int = 77
    vocab_size: int = 50257
    transformer_width: int = 512
    transformer_heads: int = 32
    transformer_layers: int = 4
    
    # Vision encoder
    vision_patch_size: int = 16
    vision_input_resolution: int = 224
    
    # Loss configuration
    loss_version: int = 0
    normalize_logits: bool = True
    log_variance: bool = False
    
    # Training dimensions (3, 3.5, 4, 6, etc.)
    dimensions: float = 6.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.dimensions not in [3, 3.5, 4, 6, -1, 0]:
            raise ValueError(f"Unsupported dimensions: {self.dimensions}")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Optimization
    learning_rate: float = 2e-3
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    total_steps: int = 200000
    
    # Batch sizes
    train_batch_size: int = 64
    eval_batch_size: int = 32
    
    # Training settings
    max_epochs: int = 20
    precision: int = 16
    gradient_clip_val: float = 0.25
    accumulate_grad_batches: int = 16
    
    # Device settings
    devices: str = "auto"
    accelerator: str = "auto"
    
    # Debugging
    fast_dev_run: bool = False
    debug: bool = False


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    # Paths
    cache_dir: str = "."
    annotations_dir: Optional[str] = None
    data_dir: str = "."
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True
    
    # Dataset specific
    num_imgs_per_val_class: int = 50
    image_size: int = 224
    
    # Language settings
    chinese_mode: bool = False
    tokenizer_name: str = "bert-base-chinese"


@dataclass
class ExperimentConfig:
    """Main experiment configuration combining all sub-configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment metadata
    experiment_name: str = "6dim_clip_experiment"
    project_name: str = "6DIMCLIPSweep"
    entity: str = "st7ma784"
    
    # Logging
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "experiment_name": self.experiment_name,
            "project_name": self.project_name,
            "entity": self.entity,
            "seed": self.seed
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            experiment_name=config_dict.get("experiment_name", "6dim_clip_experiment"),
            project_name=config_dict.get("project_name", "6DIMCLIPSweep"),
            entity=config_dict.get("entity", "st7ma784"),
            seed=config_dict.get("seed", 42)
        )


def get_device_config() -> Dict[str, Any]:
    """Get optimal device configuration based on available hardware."""
    device_config = {
        "accelerator": "auto",
        "devices": "auto",
        "precision": 16
    }
    
    if torch.cuda.is_available():
        # Check for specific GPU types that don't support certain precisions
        gpu_name = torch.cuda.get_device_name(0)
        if "P100" in gpu_name:
            device_config["precision"] = 16  # P100 doesn't support bf16
    
    return device_config
