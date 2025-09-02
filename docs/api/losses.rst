Losses Module
=============

Base Loss Classes
-----------------

.. automodule:: src.losses.base_loss
   :members:
   :undoc-members:
   :show-inheritance:

N-Dimensional Losses
--------------------

.. automodule:: src.losses.ndim_losses
   :members:
   :undoc-members:
   :show-inheritance:

Loss Factory
------------

.. automodule:: src.losses.loss_factory
   :members:
   :undoc-members:
   :show-inheritance:

CuPy Accelerated Losses
-----------------------

.. automodule:: src.losses.cupy_losses
   :members:
   :undoc-members:
   :show-inheritance:

Available Loss Functions
------------------------

The following loss functions are available:

* **EinsumLoss**: Einstein summation-based loss for n-dimensional embeddings
* **EuclideanDistanceLoss**: Euclidean distance-based contrastive loss
* **NormBasedLoss**: Norm-based loss with configurable p-norm
* **CosineSimilarityLoss**: Cosine similarity-based contrastive loss
* **CuPyEinsumLoss**: GPU-accelerated einsum loss using CuPy
* **CuPyNormBasedLoss**: GPU-accelerated norm-based loss using CuPy

Mathematical Formulations
-------------------------

Each loss function implements specific mathematical formulations optimized for multi-dimensional CLIP training. See the individual class documentation for detailed mathematical descriptions and implementation notes.
