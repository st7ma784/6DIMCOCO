6DIMCOCO: Multi-dimensional CLIP Training Framework
==================================================

Welcome to the documentation for 6DIMCOCO, a research framework for training CLIP models with novel n-dimensional loss functions and advanced analysis techniques.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   tutorials
   research
   api/index

Overview
--------

6DIMCOCO is a comprehensive framework for:

* **Multi-dimensional CLIP Training**: Train CLIP models with 3D, 4D, 6D, and other dimensional configurations
* **Novel Loss Functions**: Implement and experiment with advanced loss functions for high-dimensional embeddings
* **CKA Analysis**: Perform Centered Kernel Alignment analysis for model comparison and understanding
* **Robust Testing**: Comprehensive test suite ensuring numerical stability and correctness
* **Flexible Configuration**: Type-safe configuration management for reproducible experiments

Key Features
------------

🔬 **Research-Focused**
   Built for PhD-level research with rigorous mathematical foundations and extensive validation.

🧮 **Numerically Stable**
   All loss functions include numerical stability checks and proper gradient flow validation.

🔧 **Highly Configurable**
   Comprehensive configuration system supporting various model architectures and training setups.

📊 **Advanced Analysis**
   Built-in CKA analysis tools for deep model understanding and comparison.

🧪 **Thoroughly Tested**
   Extensive test suite covering edge cases, numerical stability, and mathematical properties.

Quick Start
-----------

.. code-block:: python

   from src.config.base_config import ExperimentConfig
   from src.losses import create_loss_function
   
   # Create experiment configuration
   config = ExperimentConfig()
   config.model.dimensions = 6.0
   config.training.learning_rate = 2e-3
   
   # Create loss function
   loss_fn = create_loss_function('norm_based', config=config.model)
   
   # Use in training...

Research Applications
--------------------

This framework has been used for research in:

* Multi-dimensional contrastive learning
* Cross-modal representation learning
* Model architecture analysis via CKA
* Numerical optimization in deep learning
* Chinese-English translation tasks

Citation
--------

If you use this framework in your research, please cite:

.. code-block:: bibtex

   @misc{6dimcoco2024,
     title={6DIMCOCO: Multi-dimensional CLIP Training Framework},
     author={PhD Research Project},
     year={2024},
     url={https://github.com/st7ma784/6DIMCOCO}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
