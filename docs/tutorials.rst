Tutorials
=========

Getting Started
---------------

This section provides step-by-step tutorials for using the 6DIMCOCO framework.

Basic Training Tutorial
-----------------------

1. **Setup Environment**

   .. code-block:: bash

      git clone https://github.com/st7ma784/6DIMCOCO.git
      cd 6DIMCOCO
      pip install -r requirements.txt

2. **Configure Your Experiment**

   .. code-block:: python

      from src.config.base_config import ExperimentConfig
      
      config = ExperimentConfig()
      config.model.dimensions = 6.0
      config.training.learning_rate = 2e-3
      config.training.batch_size = 16

3. **Run Training**

   .. code-block:: bash

      python scripts/run_training.py --wandb

Loss Function Tutorial
----------------------

1. **Using Built-in Loss Functions**

   .. code-block:: python

      from src.losses import create_loss_function
      
      # Create a 6D einsum loss
      loss_fn = create_loss_function("einsum", dimensions=6)
      
      # Create a norm-based loss
      loss_fn = create_loss_function("norm_based", p_norm=2.0)

2. **Custom Loss Functions**

   .. code-block:: python

      from src.losses.base_loss import MultiDimensionalLoss
      
      class MyCustomLoss(MultiDimensionalLoss):
          def forward(self, image_features, text_features):
              # Your custom loss implementation
              pass

CKA Analysis Tutorial
---------------------

1. **Basic CKA Analysis**

   .. code-block:: python

      from scripts.CKA_test import linear_CKA, kernel_CKA
      
      # Compare two feature representations
      cka_score = linear_CKA(features1, features2)
      print(f"CKA similarity: {cka_score:.4f}")

2. **Batch CKA Processing**

   .. code-block:: bash

      python scripts/CKA_test.py --model1 path/to/model1 --model2 path/to/model2

Dataset Building Tutorial
-------------------------

1. **ImageNet Dataset**

   .. code-block:: bash

      python data_builders/BuildImagenet.py --data_path /path/to/imagenet

2. **Custom Dataset**

   .. code-block:: python

      from data_builders.BuildCNDataset import build_chinese_dataset
      
      dataset = build_chinese_dataset(
          data_path="/path/to/data",
          split="train"
      )

Performance Optimization
------------------------

1. **Using CuPy Acceleration**

   .. code-block:: python

      from src.losses.cupy_losses import CuPyEinsumLoss
      
      # GPU-accelerated loss function
      loss_fn = CuPyEinsumLoss(dimensions=6)

2. **Benchmarking Performance**

   .. code-block:: bash

      python scripts/benchmark_cupy.py

Advanced Configuration
----------------------

1. **Multi-GPU Training**

   .. code-block:: python

      config = ExperimentConfig()
      config.training.devices = 4
      config.training.accelerator = "gpu"
      config.training.strategy = "ddp"

2. **Hyperparameter Sweeps**

   .. code-block:: python

      import wandb
      
      sweep_config = {
          'method': 'bayes',
          'parameters': {
              'learning_rate': {'min': 1e-4, 'max': 1e-2},
              'batch_size': {'values': [16, 32, 64]},
              'dimensions': {'values': [3, 4, 6]}
          }
      }

Troubleshooting
---------------

Common Issues and Solutions:

**CUDA Out of Memory**
  - Reduce batch size
  - Use gradient checkpointing
  - Enable mixed precision training

**Import Errors**
  - Ensure all dependencies are installed
  - Check Python path configuration
  - Verify CUDA installation for GPU features

**Slow Training**
  - Use CuPy acceleration for large tensors
  - Enable mixed precision (fp16)
  - Optimize data loading with multiple workers
