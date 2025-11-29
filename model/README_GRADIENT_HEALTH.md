# Gradient Health Monitoring and Numerical Stability Tools

## 🎯 Overview

This package provides comprehensive tools to diagnose and fix gradient instability in deep learning training, specifically designed for the 6-dimensional CLIP training framework. When training with high-dimensional loss functions, gradients can easily vanish, explode, or become NaN, preventing successful training.

## 🚨 Problem Statement

The 6DIMCOCO project extends CLIP to learn from 6 modalities simultaneously, but this creates challenges:

- **Computational Complexity**: Loss tensors of shape [B, B, B, B, B, B] require B^6 operations
- **Numerical Instability**: Multiple einsum operations compound numerical errors
- **Gradient Issues**: Deep computation graphs lead to vanishing/exploding gradients
- **Memory Requirements**: Large intermediate tensors can cause OOM errors

**Common symptoms:**
- ✗ Loss becomes NaN after a few steps
- ✗ Loss doesn't decrease (gradients vanishing)
- ✗ Loss explodes to infinity
- ✗ Training crashes with CUDA errors
- ✗ Model outputs become constant

## ✅ Solution

This package provides three main components:

### 1. GradientHealthCheck.py
**Real-time gradient monitoring and diagnostics**

- Monitor gradient norms across all layers
- Detect NaN/Inf immediately
- Identify vanishing/exploding gradients
- Get actionable recommendations
- Integrate with WandB/TensorBoard

### 2. StableLossFunctions.py
**Numerically stable loss implementations**

- Epsilon-protected normalization
- Clamped intermediate results
- Alternative loss formulations
- Improved activation functions
- Gradient-friendly operations

### 3. IntegrateGradientHealthChecks.py
**Easy integration with existing code**

- Mixin class for instant integration
- Automatic gradient clipping
- Activation monitoring
- Minimal code changes required

## 📦 Installation

No additional dependencies needed! All tools use standard PyTorch.

```bash
# Already installed if you have the 6DIMCOCO repository
cd /data/6DIMCOCO
```

## 🚀 Quick Start

### Option 1: Minimal Integration (5 minutes)

Add gradient monitoring to existing training code:

```python
from model.GradientHealthCheck import GradientHealthMonitor

# In your LightningModule.__init__:
self.grad_monitor = GradientHealthMonitor(
    model=self,
    grad_clip_threshold=1.0,
    log_frequency=100
)

# In on_after_backward() or after loss.backward():
stats = self.grad_monitor.check_gradients()
self.grad_monitor.log_statistics(stats, step=self.global_step)

# Apply clipping if needed:
if stats['global_grad_norm'] > 10.0:
    self.grad_monitor.apply_gradient_clipping(max_norm=1.0)
```

### Option 2: Full Integration (15 minutes)

Use the enhanced training mixin:

```python
from model.IntegrateGradientHealthChecks import EnhancedTrainingMixin

class MyModule(EnhancedTrainingMixin, LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        # Enable monitoring (one line!)
        self.setup_gradient_monitoring(
            enable_gradient_monitoring=True,
            gradient_clip_threshold=1.0
        )
    
    def on_after_backward(self):
        # Automatic gradient checking
        self.check_and_log_gradients(
            step=self.global_step,
            apply_clipping=True
        )
```

### Option 3: Use Enhanced Training Module (20 minutes)

Use the pre-built enhanced module:

```python
from model.example_enhanced_training import EnhancedLightningCLIPModule

model = EnhancedLightningCLIPModule(
    learning_rate=1e-4,
    enable_gradient_monitoring=True,
    use_stable_loss=True,
    stable_loss_version=1,  # Norm-based stable loss
    gradient_clip_value=1.0,
    # ... your other parameters ...
)
```

## 📊 Understanding the Output

When gradient issues are detected, you'll see:

```
Step 150 - Gradient Health: Global Norm=125.43, Max=340.12, Min=3.2e-08, Mean=45.67

⚠️ Exploding gradients detected (norm=125.43)!
   Consider: (1) Gradient clipping, (2) Reducing learning rate

⚠️ Vanishing gradients detected (min norm=3.2e-08)!
   Consider: (1) Using residual connections, (2) Changing activation functions
```

### What the metrics mean:

- **Global Norm**: Overall gradient magnitude (healthy: 0.1-10)
- **Max Norm**: Largest gradient in any layer (watch if >100)
- **Min Norm**: Smallest gradient (warning if <1e-7)
- **Has NaN/Inf**: CRITICAL - training will fail

## 🔧 Using Stable Loss Functions

Replace unstable loss with stable alternative:

```python
from model.StableLossFunctions import get_stable_loss_function

# Replace existing loss function
self.calculate_loss = get_stable_loss_function(
    loss_version=1,      # 0=einsum, 1=norm-based, 2=cosine, 3=distance
    normalize=True,      # Normalize inputs
    eps=1e-8            # Epsilon for numerical stability
)

# Use normally in forward pass
logits = self.calculate_loss(*features)
```

### Which loss version to use?

| Version | Method | Best For | Speed | Stability |
|---------|--------|----------|-------|-----------|
| 0 | Einsum-based | Exact computation | Slow | Medium |
| 1 | Norm-based | **General use** | Fast | **High** |
| 2 | Cosine similarity | Geometric interpretation | Medium | High |
| 3 | Distance-based | When distance matters | Fast | Medium |

**Recommendation**: Start with version 1 (norm-based) - it's the most stable.

## 🎨 Improved Activation Functions

The code currently uses various activations. Here are better alternatives:

```python
from model.StableLossFunctions import ImprovedActivations

# In your model:
self.activation = ImprovedActivations.gelu  # Smooth, non-zero gradients
# OR
self.activation = ImprovedActivations.swish  # Self-gated, deep networks
# OR
self.activation = ImprovedActivations.mish  # State-of-the-art for some tasks
```

### Activation comparison:

| Activation | Pros | Cons | Use Case |
|------------|------|------|----------|
| **GELU** | ✓ Standard in transformers<br>✓ Smooth gradients | None for most cases | **Recommended for transformers** |
| **Swish/SiLU** | ✓ Self-gated<br>✓ Outperforms ReLU | Slightly slower | Deep MLPs |
| **Mish** | ✓ Smooth<br>✓ SOTA on some tasks | Slower computation | When max performance needed |
| **LeakyReLU** | ✓ Simple<br>✓ Fast | Less smooth | Simple models |
| ReLU ❌ | Fast | Dead neurons | Avoid in deep networks |
| Tanh ❌ | Bounded | Vanishing gradients | Avoid |

## 🧪 Testing Your Setup

Run the test script to verify everything works:

```bash
python model/test_gradient_health.py
```

Expected output:
```
Testing Gradient Health Monitor
================================
✓ Gradient monitoring test passed!
  Global norm: 15.3421
  Max norm: 45.1234
  Min norm: 2.3e-05
  Has NaN: False
  Has Inf: False

Testing Stable Loss Functions
==============================
✓ Norm-based (v1):
  Shape: torch.Size([16, 16, 16])
  Range: [-2.3456, 3.4567]
  Mean: 0.1234, Std: 1.2345
  Valid: ✓
  Backward pass: ✓

ALL TESTS COMPLETED!
```

## 📈 Monitoring During Training

### WandB Integration

Gradients are automatically logged if you use WandB:

```python
# Metrics logged:
- grad/global_norm        # Overall gradient magnitude
- grad/max_norm          # Largest gradient
- grad/min_norm          # Smallest gradient
- grad/has_nan           # 0 or 1
- grad/has_inf           # 0 or 1
- grad_layer/*           # Per-layer statistics (if enabled)
```

### TensorBoard Integration

Works with any logger that has a `.log()` method.

## 🔍 Debugging Guide

### Problem: Loss is NaN after first step

**Diagnosis**: Check if gradients become NaN
```python
stats = self.grad_monitor.check_gradients()
if stats['has_nan']:
    # Gradients are NaN!
```

**Solutions** (in order):
1. Reduce learning rate by 10x
2. Add epsilon to all normalizations
3. Use stable loss function (version 1)
4. Enable gradient clipping (max_norm=1.0)
5. Check for division by zero in custom code

### Problem: Loss doesn't decrease

**Diagnosis**: Check for vanishing gradients
```python
if stats['min_grad_norm'] < 1e-7:
    # Vanishing gradients!
```

**Solutions**:
1. Change activation: ReLU → GELU
2. Add residual connections
3. Reduce network depth
4. Increase learning rate slightly
5. Check if part of model is frozen

### Problem: Loss explodes to infinity

**Diagnosis**: Check for exploding gradients
```python
if stats['global_grad_norm'] > 100:
    # Exploding gradients!
```

**Solutions**:
1. Enable gradient clipping immediately
2. Reduce learning rate by 10x
3. Add layer normalization
4. Use stable loss function
5. Reduce batch size

### Problem: Training is slow

**Solutions**:
1. Disable detailed logging (`log_frequency=1000`)
2. Disable activation monitoring
3. Use version 1 loss (fastest stable variant)
4. Check if anomaly detection is on: remove `torch.autograd.set_detect_anomaly(True)`

## 📚 Advanced Usage

### Custom Gradient Monitoring

```python
monitor = GradientHealthMonitor(
    model=self,
    grad_clip_threshold=1.0,
    vanishing_threshold=1e-7,   # Adjust for your model
    exploding_threshold=100.0,   # Adjust for your model
    log_frequency=100,
    enable_detailed_logging=True  # Per-layer stats
)

# Get detailed gradient flow analysis
flow_summary = monitor.get_gradient_flow_summary()
print(f"Layers with vanishing grads: {flow_summary['layers_with_vanishing_grads']}")
print(f"Layers with exploding grads: {flow_summary['layers_with_exploding_grads']}")
```

### Custom Stable Loss

```python
from model.StableLossFunctions import StableLossFunctions

stable_ops = StableLossFunctions(
    eps=1e-8,
    clamp_min=-100,
    clamp_max=100
)

# Build custom loss
def my_custom_loss(*features):
    # Normalize
    features = [stable_ops.stable_norm(f) for f in features]
    
    # Compute similarity
    similarity = stable_ops.stable_norm_based_similarity(*features)
    
    # Clamp result
    return torch.clamp(similarity, -10, 10)
```

### Gradient Checkpointing

For memory efficiency:

```python
from model.IntegrateGradientHealthChecks import add_gradient_checkpointing

# Add checkpointing to transformer blocks
add_gradient_checkpointing(
    model=self,
    modules_to_checkpoint=['transformer', 'ResidualAttentionBlock']
)
```

## 📖 Full Documentation

See `docs/GradientHealthGuide.md` for complete documentation including:
- Detailed explanation of the 6D cosine similarity problem
- Mathematical analysis of gradient flow
- Step-by-step integration guide
- Performance considerations
- Troubleshooting checklist

## 🤝 Contributing

If you fix a gradient issue or add improvements:
1. Document what worked in your case
2. Add test cases if possible
3. Share findings with the team

## 📞 Support

Common issues and solutions are documented in the guide. For new issues:
1. Run `python model/test_gradient_health.py` to verify setup
2. Check gradient monitor output for specific recommendations
3. Review `docs/GradientHealthGuide.md` for detailed explanations

## 📄 License

Same as parent project (see LICENSE file)

## 🙏 Acknowledgments

Built for the 6DIMCOCO project to enable stable training of high-dimensional CLIP models. Based on best practices from:
- "Understanding the difficulty of training deep feedforward neural networks" (Glorot & Bengio, 2010)
- "Searching for Activation Functions" (Ramachandran et al., 2017)
- "Mixed Precision Training" (Micikevicius et al., 2018)
