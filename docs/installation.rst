Installation
============

Requirements
------------

* Python 3.8+
* PyTorch 1.9+
* CUDA 11.0+ (optional, for GPU acceleration)

Dependencies
------------

The framework requires the following Python packages:

.. code-block:: text

   torch>=1.9.0
   torchvision>=0.10.0
   pytorch-lightning>=1.5.0
   transformers>=4.0.0
   wandb>=0.12.0
   numpy>=1.20.0
   matplotlib>=3.3.0
   scikit-learn>=1.0.0
   pytest>=6.0.0
   sphinx>=4.0.0
   sphinx-rtd-theme>=1.0.0

Installation Methods
--------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/st7ma784/6DIMCOCO.git
   cd 6DIMCOCO
   pip install -r requirements.txt
   pip install -e .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development with testing and documentation tools:

.. code-block:: bash

   git clone https://github.com/st7ma784/6DIMCOCO.git
   cd 6DIMCOCO
   pip install -r requirements-dev.txt
   pip install -e .

Docker Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker build -t 6dimcoco .
   docker run --gpus all -it 6dimcoco

Verification
------------

Verify your installation by running the test suite:

.. code-block:: bash

   pytest tests/ -v

Or run a quick functionality check:

.. code-block:: python

   from src.losses import create_loss_function
   from src.config.base_config import ModelConfig
   
   # Create a simple loss function
   config = ModelConfig()
   loss_fn = create_loss_function('stock_clip', config=config)
   print("Installation successful!")

GPU Setup
----------

For GPU acceleration, ensure CUDA is properly installed:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"GPU count: {torch.cuda.device_count()}")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Errors**
   Make sure you've installed the package in development mode with ``pip install -e .``

**CUDA Issues**
   Verify PyTorch CUDA compatibility with your system's CUDA version

**Memory Issues**
   Reduce batch sizes in configuration or use gradient accumulation

**Numerical Instability**
   Enable mixed precision training and check loss function parameters

Environment Variables
~~~~~~~~~~~~~~~~~~~~~~

Set these environment variables for optimal performance:

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specify GPUs
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   export WANDB_API_KEY=your_wandb_key  # For experiment tracking
