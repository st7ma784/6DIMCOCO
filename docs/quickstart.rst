Quick Start Guide
=================

This guide will get you up and running with 6DIMCOCO in minutes.

Basic Usage
-----------

1. **Create a Configuration**

.. code-block:: python

   from src.config.base_config import ExperimentConfig
   
   # Create default configuration
   config = ExperimentConfig()
   
   # Customize for your experiment
   config.model.dimensions = 6.0
   config.model.embed_dim = 512
   config.training.learning_rate = 2e-3
   config.training.train_batch_size = 32

2. **Create Loss Functions**

.. code-block:: python

   from src.losses import create_loss_function
   
   # Create a norm-based loss function
   loss_fn = create_loss_function('norm_based', config=config.model)
   
   # Or create from legacy version
   from src.losses import create_loss_from_legacy_version
   legacy_loss = create_loss_from_legacy_version(version=10)

3. **Use in Training**

.. code-block:: python

   import torch
   
   # Create sample features (6 tensors for 6D loss)
   batch_size = 32
   feature_dim = 512
   features = [torch.randn(batch_size, feature_dim) for _ in range(6)]
   
   # Compute loss
   loss_value = loss_fn(*features)
   print(f"Loss shape: {loss_value.shape}")

Training Example
----------------

Here's a complete training example:

.. code-block:: python

   import torch
   import torch.nn as nn
   from src.config.base_config import ExperimentConfig
   from src.losses import create_loss_function
   
   class SimpleModel(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.encoder = nn.Linear(784, config.model.embed_dim)
           self.projections = nn.ModuleList([
               nn.Linear(config.model.embed_dim, config.model.embed_dim) 
               for _ in range(6)
           ])
       
       def forward(self, x):
           features = self.encoder(x)
           projections = [proj(features) for proj in self.projections]
           return projections
   
   # Setup
   config = ExperimentConfig()
   model = SimpleModel(config)
   loss_fn = create_loss_function('norm_based', config=config.model)
   optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
   
   # Training loop
   for epoch in range(10):
       # Dummy data
       x = torch.randn(config.training.train_batch_size, 784)
       
       # Forward pass
       features = model(x)
       loss = loss_fn(*features).mean()
       
       # Backward pass
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       
       print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

Available Loss Functions
------------------------

.. code-block:: python

   from src.losses import get_available_losses
   
   # See all available loss functions
   losses = get_available_losses()
   for name, description in losses.items():
       print(f"{name}: {description}")

Output:

.. code-block:: text

   stock_clip: Standard CLIP contrastive loss
   einsum: Einstein summation based n-dimensional loss
   euclidean_distance: Euclidean distance based loss with stability
   norm_based: Norm-based loss with multiple variants
   cosine_similarity: Cosine similarity based multi-dimensional loss

Configuration Options
---------------------

**Model Configuration:**

.. code-block:: python

   config.model.dimensions = 6.0        # 3, 3.5, 4, 6, -1, 0
   config.model.embed_dim = 512         # Embedding dimension
   config.model.normalize_logits = True # Normalize features
   config.model.loss_version = 0        # Legacy loss version

**Training Configuration:**

.. code-block:: python

   config.training.learning_rate = 2e-3
   config.training.train_batch_size = 64
   config.training.max_epochs = 20
   config.training.precision = 16       # Mixed precision

**Data Configuration:**

.. code-block:: python

   config.data.cache_dir = "./data"
   config.data.chinese_mode = False     # Enable Chinese language support
   config.data.image_size = 224

Testing Your Setup
------------------

Run the test suite to verify everything works:

.. code-block:: bash

   # Run all tests
   pytest tests/ -v
   
   # Run only loss function tests
   pytest tests/test_losses.py -v
   
   # Run without GPU tests
   pytest tests/ -m "not gpu" -v

CKA Analysis Example
--------------------

.. code-block:: python

   import torch
   from tests.test_cka_analysis import TestCKAMethods
   
   # Create sample kernel matrices
   batch_size = 50
   K = torch.randn(batch_size, batch_size)
   K = K @ K.T  # Make positive semi-definite
   K.fill_diagonal_(0)
   
   L = torch.randn(batch_size, batch_size)
   L = L @ L.T
   L.fill_diagonal_(0)
   
   # Compute HSIC (used in CKA)
   def compute_hsic(K, L):
       N = K.shape[0]
       trace_term = torch.trace(K @ L)
       sum_K = torch.sum(K)
       sum_L = torch.sum(L)
       sum_KL = torch.sum(K @ L)
       
       correction = (sum_K * sum_L / (N - 1) - sum_KL * 2) / (N - 2)
       return trace_term + correction
   
   hsic_value = compute_hsic(K, L)
   print(f"HSIC value: {hsic_value}")

Next Steps
----------

For more detailed information, see:

* :doc:`tutorials` - Step-by-step tutorials
* :doc:`research` - Research applications and theory
* :doc:`api/index` - Complete API reference
* Run experiments with different loss functions and configurations
