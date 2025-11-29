"""
Example Integration of Gradient Health Checks and Stable Loss Functions

This script demonstrates how to integrate the gradient monitoring and
stable loss functions into existing training code.

Usage:
    from model.IntegrateGradientHealthChecks import integrate_health_checks
    
    # In your training module
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        return integrate_health_checks(self, loss, batch_idx)
"""

import torch
import torch.nn as nn
from model.GradientHealthCheck import GradientHealthMonitor, ActivationMonitor
from model.StableLossFunctions import get_stable_loss_function, StableLossFunctions
from typing import Dict, Optional
import logging


class EnhancedTrainingMixin:
    """
    Mixin class to add gradient health monitoring to existing training modules.
    
    Add this to your LightningModule class:
        class MyModule(EnhancedTrainingMixin, LightningModule):
            def __init__(self, ...):
                super().__init__(...)
                self.setup_gradient_monitoring()
    """
    
    def setup_gradient_monitoring(
        self,
        enable_gradient_monitoring: bool = True,
        enable_activation_monitoring: bool = False,
        gradient_clip_threshold: float = 1.0,
        log_frequency: int = 100
    ):
        """
        Setup gradient and activation monitoring.
        
        Args:
            enable_gradient_monitoring: Whether to monitor gradients
            enable_activation_monitoring: Whether to monitor activations
            gradient_clip_threshold: Threshold for gradient clipping
            log_frequency: How often to log statistics
        """
        self.enable_gradient_monitoring = enable_gradient_monitoring
        self.enable_activation_monitoring = enable_activation_monitoring
        
        if enable_gradient_monitoring:
            self.gradient_monitor = GradientHealthMonitor(
                model=self,
                grad_clip_threshold=gradient_clip_threshold,
                vanishing_threshold=1e-7,
                exploding_threshold=100.0,
                log_frequency=log_frequency
            )
            print("✓ Gradient health monitoring enabled")
        
        if enable_activation_monitoring:
            self.activation_monitor = ActivationMonitor(model=self)
            self.activation_monitor.register_hooks()
            print("✓ Activation monitoring enabled")
    
    def check_and_log_gradients(
        self,
        step: int,
        apply_clipping: bool = False,
        max_grad_norm: float = 1.0
    ) -> Dict:
        """
        Check gradient health and optionally apply clipping.
        
        Args:
            step: Current training step
            apply_clipping: Whether to apply gradient clipping
            max_grad_norm: Maximum gradient norm for clipping
        
        Returns:
            Dictionary of gradient statistics
        """
        if not self.enable_gradient_monitoring:
            return {}
        
        # Check gradients
        stats = self.gradient_monitor.check_gradients()
        
        # Log to wandb or other logger if available
        logger = getattr(self, 'logger', None)
        if logger and hasattr(logger, 'experiment'):
            self.gradient_monitor.log_statistics(stats, step=step, logger=logger.experiment)
        else:
            self.gradient_monitor.log_statistics(stats, step=step)
        
        # Apply gradient clipping if requested or if exploding gradients detected
        if apply_clipping or stats.get('global_grad_norm', 0) > 100:
            actual_norm = self.gradient_monitor.apply_gradient_clipping(max_norm=max_grad_norm)
            if hasattr(self, 'log'):
                self.log('grad_norm_before_clip', actual_norm, prog_bar=False)
                self.log('grad_norm_after_clip', max_grad_norm, prog_bar=False)
        
        return stats
    
    def get_activation_statistics(self) -> Dict:
        """Get activation statistics if monitoring is enabled."""
        if not self.enable_activation_monitoring:
            return {}
        
        stats = self.activation_monitor.get_statistics()
        dead_neurons = self.activation_monitor.detect_dead_neurons()
        
        if dead_neurons:
            logging.warning(f"Detected dead neurons in {len(dead_neurons)} layers:")
            for layer, sparsity in dead_neurons.items():
                logging.warning(f"  {layer}: {sparsity:.2%} sparsity")
        
        return {"activations": stats, "dead_neurons": dead_neurons}


def integrate_health_checks(
    module,
    loss: torch.Tensor,
    step: int,
    apply_gradient_clipping: bool = True,
    max_grad_norm: float = 1.0
) -> Dict:
    """
    Integrate gradient health checks into training step.
    
    Call this AFTER loss.backward() but BEFORE optimizer.step().
    
    Args:
        module: Your training module (must have gradient_monitor attribute)
        loss: The computed loss tensor
        step: Current training step
        apply_gradient_clipping: Whether to apply gradient clipping
        max_grad_norm: Maximum gradient norm
    
    Returns:
        Dictionary with loss and gradient statistics
    """
    result = {"loss": loss}
    
    # Check if module has gradient monitoring
    if hasattr(module, 'gradient_monitor'):
        grad_stats = module.check_and_log_gradients(
            step=step,
            apply_clipping=apply_gradient_clipping,
            max_grad_norm=max_grad_norm
        )
        result["gradient_stats"] = grad_stats
        
        # Log critical metrics
        if hasattr(module, 'log'):
            module.log('grad/global_norm', grad_stats.get('global_grad_norm', 0))
            module.log('grad/has_nan', float(grad_stats.get('has_nan', False)))
            module.log('grad/has_inf', float(grad_stats.get('has_inf', False)))
    
    return result


def replace_loss_function_with_stable_version(
    module,
    loss_version: int = 1,
    normalize: bool = True,
    eps: float = 1e-8
):
    """
    Replace the existing loss function with a stable version.
    
    Args:
        module: Your training module
        loss_version: Which stable loss variant to use
        normalize: Whether to normalize inputs
        eps: Epsilon for numerical stability
    """
    stable_loss_fn = get_stable_loss_function(
        loss_version=loss_version,
        normalize=normalize,
        eps=eps
    )
    
    # Store original loss function as backup
    if hasattr(module, 'calculate_loss'):
        module.calculate_loss_original = module.calculate_loss
    
    # Replace with stable version
    module.calculate_loss = stable_loss_fn
    
    print(f"✓ Replaced loss function with stable version {loss_version}")
    print(f"  - Normalization: {normalize}")
    print(f"  - Epsilon: {eps}")


def add_gradient_checkpointing(model: nn.Module, modules_to_checkpoint: list = None):
    """
    Add gradient checkpointing to reduce memory usage and improve stability.
    
    Args:
        model: The model to add checkpointing to
        modules_to_checkpoint: List of module names or types to checkpoint
                              If None, checkpoint transformer blocks
    """
    from torch.utils.checkpoint import checkpoint
    
    if modules_to_checkpoint is None:
        # Default: checkpoint transformer blocks
        modules_to_checkpoint = ['transformer', 'TransformerEncoderLayer', 'ResidualAttentionBlock']
    
    def create_checkpoint_forward(original_forward):
        def forward(*args, **kwargs):
            return checkpoint(original_forward, *args, **kwargs)
        return forward
    
    checkpoint_count = 0
    for name, module in model.named_modules():
        if any(target in name or target in type(module).__name__ 
               for target in modules_to_checkpoint):
            # Wrap the forward method with checkpointing
            module.forward = create_checkpoint_forward(module.forward)
            checkpoint_count += 1
    
    print(f"✓ Added gradient checkpointing to {checkpoint_count} modules")


# Example usage in a Lightning module
EXAMPLE_USAGE = """
# In your LightningCLIPModule class:

from model.IntegrateGradientHealthChecks import (
    EnhancedTrainingMixin,
    replace_loss_function_with_stable_version,
    add_gradient_checkpointing
)

class LightningCLIPModule(EnhancedTrainingMixin, LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        # ... your existing initialization ...
        
        # Setup gradient monitoring
        self.setup_gradient_monitoring(
            enable_gradient_monitoring=True,
            enable_activation_monitoring=False,
            gradient_clip_threshold=1.0,
            log_frequency=100
        )
        
        # Replace loss function with stable version
        replace_loss_function_with_stable_version(
            self,
            loss_version=1,  # Use norm-based stable loss
            normalize=True,
            eps=1e-8
        )
        
        # Optional: Add gradient checkpointing for memory efficiency
        # add_gradient_checkpointing(self)
    
    def training_step(self, batch, batch_idx):
        # ... your existing training code ...
        
        logits = self(im, *captions) * self.logit_scale.exp()
        
        # Compute loss
        losses = [self.loss(logits.permute(*i), labels, alpha=self.alpha) 
                  for i in permutes]
        loss = self.meanloss(I=[losses[0]], T=losses[1:]).mean()
        
        # After backward is called by Lightning, gradients will be checked
        # But you can manually check here too:
        # loss.backward()  # If not using Lightning's automatic backward
        # grad_stats = self.check_and_log_gradients(
        #     step=batch_idx,
        #     apply_clipping=True,
        #     max_grad_norm=1.0
        # )
        
        self.log('train_loss', loss, prog_bar=True)
        
        return {"loss": loss}
    
    def on_after_backward(self):
        '''Hook called after backward pass - perfect place for gradient checks'''
        if self.enable_gradient_monitoring:
            self.check_and_log_gradients(
                step=self.global_step,
                apply_clipping=True,
                max_grad_norm=1.0
            )
"""

print(__doc__)
print("\n" + "="*70)
print("EXAMPLE USAGE:")
print("="*70)
print(EXAMPLE_USAGE)
