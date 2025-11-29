"""
Gradient Health Check Utilities for Multi-dimensional CLIP Training

This module provides comprehensive gradient monitoring and debugging tools
to help diagnose unstable training in high-dimensional loss functions.

Key Features:
- Gradient norm tracking across all parameters
- NaN/Inf detection and reporting
- Per-layer gradient statistics
- Activation statistics monitoring
- Automatic gradient clipping recommendations
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict
import numpy as np


class GradientHealthMonitor:
    """
    Monitor gradient health during training to detect vanishing/exploding gradients.
    
    Usage:
        monitor = GradientHealthMonitor(model)
        
        # In training loop:
        loss.backward()
        health_stats = monitor.check_gradients()
        monitor.log_statistics(health_stats, step=batch_idx)
    """
    
    def __init__(
        self,
        model: nn.Module,
        grad_clip_threshold: float = 1.0,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 100.0,
        log_frequency: int = 100,
        enable_detailed_logging: bool = True
    ):
        """
        Args:
            model: The PyTorch model to monitor
            grad_clip_threshold: Threshold for gradient clipping recommendation
            vanishing_threshold: Threshold below which gradients are considered vanishing
            exploding_threshold: Threshold above which gradients are considered exploding
            log_frequency: How often to log detailed statistics
            enable_detailed_logging: Whether to log per-layer statistics
        """
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.log_frequency = log_frequency
        self.enable_detailed_logging = enable_detailed_logging
        
        self.step_counter = 0
        self.gradient_history = defaultdict(list)
        self.nan_inf_counts = {"nan": 0, "inf": 0}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def check_gradients(self) -> Dict:
        """
        Check gradient health and return statistics.
        
        Returns:
            Dictionary containing gradient statistics:
            - global_grad_norm: Overall gradient norm
            - max_grad_norm: Maximum gradient norm across layers
            - min_grad_norm: Minimum gradient norm across layers
            - has_nan: Whether any gradients are NaN
            - has_inf: Whether any gradients are Inf
            - layer_stats: Per-layer gradient statistics
            - recommendations: List of recommended actions
        """
        stats = {
            "global_grad_norm": 0.0,
            "max_grad_norm": 0.0,
            "min_grad_norm": float('inf'),
            "mean_grad_norm": 0.0,
            "has_nan": False,
            "has_inf": False,
            "layer_stats": {},
            "recommendations": [],
            "param_count": 0,
            "params_with_grads": 0
        }
        
        total_norm = 0.0
        layer_norms = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                stats["params_with_grads"] += 1
                
                # Check for NaN/Inf
                if torch.isnan(param.grad).any():
                    stats["has_nan"] = True
                    self.nan_inf_counts["nan"] += 1
                    self.logger.error(f"NaN detected in gradients of {name}")
                
                if torch.isinf(param.grad).any():
                    stats["has_inf"] = True
                    self.nan_inf_counts["inf"] += 1
                    self.logger.error(f"Inf detected in gradients of {name}")
                
                # Compute norms
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                layer_norms.append(param_norm)
                
                # Per-layer statistics
                stats["layer_stats"][name] = {
                    "norm": param_norm,
                    "mean": param.grad.data.mean().item(),
                    "std": param.grad.data.std().item(),
                    "max": param.grad.data.max().item(),
                    "min": param.grad.data.min().item(),
                    "shape": list(param.grad.shape)
                }
                
                # Update extremes
                stats["max_grad_norm"] = max(stats["max_grad_norm"], param_norm)
                stats["min_grad_norm"] = min(stats["min_grad_norm"], param_norm)
            
            stats["param_count"] += 1
        
        # Compute global norm
        if layer_norms:
            stats["global_grad_norm"] = np.sqrt(total_norm)
            stats["mean_grad_norm"] = np.mean(layer_norms)
        
        # Generate recommendations
        stats["recommendations"] = self._generate_recommendations(stats)
        
        # Update history
        self.gradient_history["global_norm"].append(stats["global_grad_norm"])
        self.gradient_history["max_norm"].append(stats["max_grad_norm"])
        self.gradient_history["min_norm"].append(stats["min_grad_norm"])
        
        self.step_counter += 1
        
        return stats
    
    def _generate_recommendations(self, stats: Dict) -> List[str]:
        """Generate recommendations based on gradient statistics."""
        recommendations = []
        
        if stats["has_nan"]:
            recommendations.append(
                "⚠️  CRITICAL: NaN detected in gradients! "
                "Consider: (1) Reducing learning rate, "
                "(2) Adding gradient clipping, "
                "(3) Checking loss function stability, "
                "(4) Using mixed precision with loss scaling"
            )
        
        if stats["has_inf"]:
            recommendations.append(
                "⚠️  CRITICAL: Inf detected in gradients! "
                "This usually indicates numerical overflow. "
                "Consider: (1) Reducing learning rate dramatically, "
                "(2) Adding normalization layers, "
                "(3) Checking for division by zero in loss"
            )
        
        if stats["global_grad_norm"] > self.exploding_threshold:
            recommendations.append(
                f"⚠️  Exploding gradients detected (norm={stats['global_grad_norm']:.2f})! "
                f"Consider: (1) Gradient clipping (torch.nn.utils.clip_grad_norm_), "
                f"(2) Reducing learning rate, "
                f"(3) Using gradient accumulation"
            )
        
        if stats["min_grad_norm"] < self.vanishing_threshold:
            recommendations.append(
                f"⚠️  Vanishing gradients detected (min norm={stats['min_grad_norm']:.2e})! "
                f"Consider: (1) Using residual connections, "
                f"(2) Changing activation functions (e.g., ReLU→LeakyReLU/GELU), "
                f"(3) Batch normalization or Layer normalization, "
                f"(4) Increasing learning rate for affected layers"
            )
        
        # Check for dead parameters
        if stats["params_with_grads"] < stats["param_count"]:
            dead_params = stats["param_count"] - stats["params_with_grads"]
            recommendations.append(
                f"⚠️  {dead_params} parameters have no gradients! "
                f"Check if parts of your model are frozen or not receiving gradients."
            )
        
        return recommendations
    
    def log_statistics(self, stats: Dict, step: Optional[int] = None, logger=None):
        """
        Log gradient statistics.
        
        Args:
            stats: Statistics dictionary from check_gradients()
            step: Current training step
            logger: Optional external logger (e.g., wandb, tensorboard)
        """
        step = step if step is not None else self.step_counter
        
        # Always log critical issues
        if stats["has_nan"] or stats["has_inf"]:
            self.logger.error(f"Step {step}: CRITICAL gradient issues detected!")
            for rec in stats["recommendations"]:
                self.logger.error(rec)
        
        # Log summary statistics
        if step % self.log_frequency == 0:
            self.logger.info(
                f"Step {step} - Gradient Health: "
                f"Global Norm={stats['global_grad_norm']:.4f}, "
                f"Max={stats['max_grad_norm']:.4f}, "
                f"Min={stats['min_grad_norm']:.2e}, "
                f"Mean={stats['mean_grad_norm']:.4f}"
            )
            
            # Log recommendations if any
            if stats["recommendations"]:
                self.logger.warning("Recommendations:")
                for rec in stats["recommendations"]:
                    self.logger.warning(f"  {rec}")
        
        # Log to external logger if provided
        if logger is not None:
            if hasattr(logger, 'log'):  # wandb-style
                logger.log({
                    "grad/global_norm": stats["global_grad_norm"],
                    "grad/max_norm": stats["max_grad_norm"],
                    "grad/min_norm": stats["min_grad_norm"],
                    "grad/mean_norm": stats["mean_grad_norm"],
                    "grad/has_nan": int(stats["has_nan"]),
                    "grad/has_inf": int(stats["has_inf"]),
                }, step=step)
                
                # Log per-layer statistics if enabled
                if self.enable_detailed_logging:
                    for layer_name, layer_stats in stats["layer_stats"].items():
                        # Clean up layer name for logging
                        clean_name = layer_name.replace('.', '/')
                        logger.log({
                            f"grad_layer/{clean_name}/norm": layer_stats["norm"],
                            f"grad_layer/{clean_name}/mean": layer_stats["mean"],
                            f"grad_layer/{clean_name}/std": layer_stats["std"],
                        }, step=step)
    
    def apply_gradient_clipping(self, max_norm: float = 1.0):
        """
        Apply gradient clipping to prevent exploding gradients.
        
        Args:
            max_norm: Maximum norm of the gradients
        
        Returns:
            Total norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=max_norm
        )
    
    def get_gradient_flow_summary(self) -> Dict:
        """
        Get a summary of gradient flow through the network.
        Useful for identifying which layers have gradient issues.
        """
        summary = {
            "layers_with_vanishing_grads": [],
            "layers_with_exploding_grads": [],
            "layers_with_healthy_grads": [],
        }
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                
                if grad_norm < self.vanishing_threshold:
                    summary["layers_with_vanishing_grads"].append(
                        (name, grad_norm)
                    )
                elif grad_norm > self.exploding_threshold:
                    summary["layers_with_exploding_grads"].append(
                        (name, grad_norm)
                    )
                else:
                    summary["layers_with_healthy_grads"].append(
                        (name, grad_norm)
                    )
        
        return summary
    
    def reset_statistics(self):
        """Reset accumulated statistics."""
        self.gradient_history.clear()
        self.nan_inf_counts = {"nan": 0, "inf": 0}
        self.step_counter = 0


class ActivationMonitor:
    """
    Monitor activations during forward pass to detect saturation or dead neurons.
    
    Usage:
        monitor = ActivationMonitor(model)
        monitor.register_hooks()
        
        # During training
        output = model(input)
        stats = monitor.get_statistics()
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = {}
        self.hooks = []
    
    def _hook_fn(self, name):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = {
                    "mean": output.data.mean().item(),
                    "std": output.data.std().item(),
                    "max": output.data.max().item(),
                    "min": output.data.min().item(),
                    "sparsity": (output.data.abs() < 1e-6).float().mean().item(),
                    "shape": list(output.shape)
                }
        return hook
    
    def register_hooks(self):
        """Register forward hooks on all modules."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_statistics(self) -> Dict:
        """Get activation statistics."""
        return self.activations.copy()
    
    def detect_dead_neurons(self, threshold: float = 1e-6) -> Dict:
        """
        Detect layers with potentially dead neurons.
        
        Args:
            threshold: Activation threshold below which neurons are considered dead
        
        Returns:
            Dictionary of layers with high sparsity
        """
        dead_neurons = {}
        for name, stats in self.activations.items():
            if stats["sparsity"] > 0.5:  # More than 50% near-zero activations
                dead_neurons[name] = stats["sparsity"]
        return dead_neurons


def add_gradient_noise(
    model: nn.Module,
    noise_std: float = 1e-5,
    gamma: float = 0.55
) -> None:
    """
    Add gradient noise to help escape sharp minima and improve stability.
    
    This implements gradient noise injection as described in:
    "Adding Gradient Noise Improves Learning for Very Deep Networks"
    
    Args:
        model: The model to add noise to
        noise_std: Standard deviation of noise
        gamma: Annealing rate (noise_std * (1 + step)^(-gamma))
    """
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_std
            param.grad.add_(noise)


def get_activation_function_recommendations(
    layer_type: str,
    problem: str = "vanishing"
) -> List[str]:
    """
    Get activation function recommendations based on layer type and problem.
    
    Args:
        layer_type: Type of layer ("transformer", "mlp", "projection", etc.)
        problem: Type of problem ("vanishing", "exploding", "dead_neurons")
    
    Returns:
        List of recommended activation functions with rationale
    """
    recommendations = []
    
    if problem == "vanishing":
        recommendations.extend([
            "✓ GELU: Smooth, non-zero gradients everywhere. Excellent for transformers.",
            "✓ Swish/SiLU: Self-gated, smooth. Often better than ReLU for deep networks.",
            "✓ LeakyReLU/ELU: Non-zero gradients for negative inputs.",
            "✓ Mish: Smooth, unbounded above, bounded below. Good for deep networks.",
        ])
    
    elif problem == "exploding":
        recommendations.extend([
            "✓ Tanh: Bounded output [-1, 1]. Natural saturation.",
            "✓ Sigmoid: Bounded output [0, 1]. Use with caution in deep networks.",
            "✓ Hard Swish: Faster than Swish, bounded at extremes.",
            "✓ Consider adding Layer Normalization before activation.",
        ])
    
    elif problem == "dead_neurons":
        recommendations.extend([
            "✓ LeakyReLU: Prevents dying ReLU problem.",
            "✓ ELU: Smooth, negative values have gradients.",
            "✓ SELU: Self-normalizing, good for deep MLPs.",
            "✓ Swish/SiLU: No hard zero, always has gradients.",
        ])
    
    # Layer-specific recommendations
    if layer_type == "transformer":
        recommendations.append(
            "🔧 For transformers: GELU is standard and works well. "
            "Consider GeLU→Swish if vanishing gradients persist."
        )
    elif layer_type == "projection":
        recommendations.append(
            "🔧 For projection layers: Often no activation is best, "
            "or use linear projection with layer norm."
        )
    
    return recommendations
