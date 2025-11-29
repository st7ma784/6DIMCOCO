"""
Example: Enhanced trainclip_v534DIM with Gradient Health Checks

This file demonstrates how to modify the existing training code to include:
1. Gradient health monitoring
2. Stable loss functions
3. Better activation functions

This is a reference implementation - adapt to your specific needs.
"""

from functools import reduce
from operator import add
from model.trainclip_v5335DIM import LightningCLIPModule as base 
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, CLIPTokenizer
import numpy as np

# Import new utilities
from model.GradientHealthCheck import GradientHealthMonitor, ActivationMonitor
from model.StableLossFunctions import get_stable_loss_function, ImprovedActivations
from model.IntegrateGradientHealthChecks import EnhancedTrainingMixin


class EnhancedLightningCLIPModule(EnhancedTrainingMixin, base):
    """
    Enhanced version of LightningCLIPModule with gradient health monitoring
    and numerical stability improvements.
    
    Key additions:
    - Gradient health monitoring
    - Stable loss function option
    - Improved activation functions
    - Automatic gradient clipping
    - Better numerical stability
    """
    
    def __init__(self, *args, **kwargs):
        # Extract new parameters
        enable_gradient_monitoring = kwargs.pop('enable_gradient_monitoring', True)
        enable_activation_monitoring = kwargs.pop('enable_activation_monitoring', False)
        use_stable_loss = kwargs.pop('use_stable_loss', True)
        stable_loss_version = kwargs.pop('stable_loss_version', 1)
        gradient_clip_value = kwargs.pop('gradient_clip_value', 1.0)
        improved_activation = kwargs.pop('improved_activation', 'gelu')
        
        # Call parent constructor
        super().__init__(*args, **kwargs)
        
        # Setup gradient monitoring
        if enable_gradient_monitoring:
            self.setup_gradient_monitoring(
                enable_gradient_monitoring=True,
                enable_activation_monitoring=enable_activation_monitoring,
                gradient_clip_threshold=gradient_clip_value,
                log_frequency=100
            )
            print("✓ Gradient health monitoring enabled")
        else:
            self.enable_gradient_monitoring = False
        
        # Store gradient clip value for later use
        self.gradient_clip_value = gradient_clip_value
        
        # Replace loss function with stable version if requested
        if use_stable_loss:
            print(f"✓ Using stable loss function (version {stable_loss_version})")
            self.calculate_loss_original = self.calculate_loss
            self.calculate_loss = get_stable_loss_function(
                loss_version=stable_loss_version,
                normalize=self.hparams.normlogits if hasattr(self.hparams, 'normlogits') else True,
                eps=1e-8
            )
        
        # Store activation function
        self.improved_activation = improved_activation
        
        # Initialize statistics tracking
        self.gradient_stats_history = []
        self.last_grad_check_step = -1
    
    def training_step(self, batch, batch_idx):
        """Enhanced training step with gradient monitoring."""
        
        im, captions = batch[0], batch[1]
        assert len(self.label.shape) >= 4
        
        # Forward pass
        logits = self(im, *[captions[:, i] for i in range(captions.shape[1])]) * self.logit_scale.exp()
        
        try:
            labels = self.label[:(im.shape[0]), :(im.shape[0]), :(im.shape[0]), :(im.shape[0])].to(
                self.device, non_blocking=True
            )
        except:
            labels = self.generate_labels((len(logits.shape), self.hparams.batch_size, self.transformer_width)).to(
                self.device, non_blocking=True
            )
        
        # Log some sample logits
        self.log("first logit", logits[0, 0, 0, 0], enable_graph=False)
        self.log("BAD logit", logits[0, 1, 2, 3], enable_graph=False)
        self.log("logit scale", self.logit_scale.exp())
        
        # Check for NaN/Inf in logits before computing loss
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            self.logger.experiment.log({
                'logits_has_nan': torch.isnan(logits).any().item(),
                'logits_has_inf': torch.isinf(logits).any().item(),
            })
            print("⚠️ WARNING: NaN or Inf detected in logits!")
        
        # Compute losses
        n_dims = len(logits.shape)
        dims = np.arange(n_dims).repeat(n_dims).reshape(n_dims, n_dims)
        dims_ = np.arange(n_dims)
        dims_ = np.expand_dims(dims_, axis=0)
        permutes = dims + dims_
        permutes = permutes % n_dims
        
        losses = [self.loss(logits.permute(*i), labels, alpha=self.alpha) for i in permutes]
        loss = self.meanloss(I=[losses[0]], T=losses[1:]).mean()
        
        # Log loss statistics
        self.log('train_loss', loss, prog_bar=True, enable_graph=False, rank_zero_only=True)
        
        # Additional loss health checks
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.experiment.log({
                'loss_is_nan': torch.isnan(loss).item(),
                'loss_is_inf': torch.isinf(loss).item(),
            })
            print("⚠️ CRITICAL: NaN or Inf detected in loss!")
        
        return {"loss": loss}
    
    def on_after_backward(self):
        """
        Hook called after backward pass - perfect place for gradient checks.
        This is called automatically by PyTorch Lightning.
        """
        if not self.enable_gradient_monitoring:
            return
        
        # Only check every N steps to reduce overhead
        if self.global_step % 10 == 0:
            # Check gradient health
            stats = self.check_and_log_gradients(
                step=self.global_step,
                apply_clipping=True,
                max_grad_norm=self.gradient_clip_value
            )
            
            # Store stats for analysis
            self.gradient_stats_history.append({
                'step': self.global_step,
                'global_norm': stats.get('global_grad_norm', 0),
                'max_norm': stats.get('max_grad_norm', 0),
                'min_norm': stats.get('min_grad_norm', 0),
                'has_nan': stats.get('has_nan', False),
                'has_inf': stats.get('has_inf', False),
            })
            
            # Log to wandb if available
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({
                    'grad/global_norm': stats.get('global_grad_norm', 0),
                    'grad/max_norm': stats.get('max_grad_norm', 0),
                    'grad/min_norm': stats.get('min_grad_norm', 0),
                    'grad/has_nan': int(stats.get('has_nan', False)),
                    'grad/has_inf': int(stats.get('has_inf', False)),
                }, step=self.global_step)
            
            # Take action if gradients are unhealthy
            if stats.get('has_nan', False) or stats.get('has_inf', False):
                print(f"\n⚠️ CRITICAL: Gradient issues at step {self.global_step}")
                print("Recommendations:")
                for rec in stats.get('recommendations', []):
                    print(f"  - {rec}")
    
    def configure_optimizers(self):
        """
        Configure optimizers with gradient clipping support.
        """
        if self.hparams.precision == 8:
            from model.LionOptimizer import Lion as lion
            optimizer = lion(self.parameters(), lr=self.hparams.learning_rate)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                eps=1e-8,
                weight_decay=0.01  # Added small weight decay for stability
            )
        
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            ),
            "monitor": "train_loss",
            "interval": "epoch",
            "frequency": 1
        }
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            # Enable gradient clipping
            'gradient_clip_val': self.gradient_clip_value,
            'gradient_clip_algorithm': 'norm'
        }
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        super().on_train_epoch_end()
        
        # Print gradient health summary
        if self.enable_gradient_monitoring and self.gradient_stats_history:
            recent_stats = self.gradient_stats_history[-100:]  # Last 100 steps
            
            avg_norm = np.mean([s['global_norm'] for s in recent_stats])
            max_norm = np.max([s['max_norm'] for s in recent_stats])
            nan_count = sum([s['has_nan'] for s in recent_stats])
            inf_count = sum([s['has_inf'] for s in recent_stats])
            
            print(f"\n📊 Gradient Health Summary (Epoch {self.current_epoch}):")
            print(f"   Average gradient norm: {avg_norm:.4f}")
            print(f"   Maximum gradient norm: {max_norm:.4f}")
            print(f"   NaN occurrences: {nan_count}")
            print(f"   Inf occurrences: {inf_count}")
            
            if nan_count > 0 or inf_count > 0:
                print("   ⚠️ Gradient health issues detected! Consider:")
                print("      - Reducing learning rate")
                print("      - Using more aggressive gradient clipping")
                print("      - Checking loss function stability")


# Usage example
if __name__ == "__main__":
    print("""
    Example Usage:
    
    from model.example_enhanced_training import EnhancedLightningCLIPModule
    
    model = EnhancedLightningCLIPModule(
        learning_rate=1e-4,  # Lower learning rate for stability
        enable_gradient_monitoring=True,
        enable_activation_monitoring=False,
        use_stable_loss=True,
        stable_loss_version=1,  # Norm-based stable loss
        gradient_clip_value=1.0,
        improved_activation='gelu',
        # ... other parameters ...
    )
    
    trainer = Trainer(
        max_epochs=100,
        # gradient_clip_val is now configured in configure_optimizers
    )
    
    trainer.fit(model, datamodule)
    """)
