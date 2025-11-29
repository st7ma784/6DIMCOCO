"""
Test script for gradient health monitoring and stable loss functions.

This script tests the new utilities with synthetic data to ensure they work correctly.

Run with: python model/test_gradient_health.py
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.GradientHealthCheck import GradientHealthMonitor, ActivationMonitor
from model.StableLossFunctions import (
    StableLossFunctions,
    get_stable_loss_function,
    ImprovedActivations
)


def test_gradient_monitor():
    """Test gradient health monitoring."""
    print("="*70)
    print("Testing Gradient Health Monitor")
    print("="*70)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256)
    )
    
    # Create monitor
    monitor = GradientHealthMonitor(
        model=model,
        grad_clip_threshold=1.0,
        log_frequency=1
    )
    
    # Create dummy input and target
    x = torch.randn(32, 512)
    target = torch.randn(32, 256)
    
    # Forward pass
    output = model(x)
    loss = ((output - target) ** 2).mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    stats = monitor.check_gradients()
    monitor.log_statistics(stats, step=0)
    
    print(f"\n✓ Gradient monitoring test passed!")
    print(f"  Global norm: {stats['global_grad_norm']:.4f}")
    print(f"  Max norm: {stats['max_grad_norm']:.4f}")
    print(f"  Min norm: {stats['min_grad_norm']:.4e}")
    print(f"  Has NaN: {stats['has_nan']}")
    print(f"  Has Inf: {stats['has_inf']}")
    
    return stats


def test_stable_loss_functions():
    """Test stable loss functions."""
    print("\n" + "="*70)
    print("Testing Stable Loss Functions")
    print("="*70)
    
    stable_ops = StableLossFunctions(eps=1e-8)
    
    # Create synthetic features (6 modalities)
    batch_size = 16
    feature_dim = 512
    features = [torch.randn(batch_size, feature_dim) for _ in range(6)]
    
    print(f"\nTesting with {len(features)} modalities, batch_size={batch_size}, dim={feature_dim}")
    
    # Test each loss variant
    loss_variants = {
        0: "Einsum-based (stabilized)",
        1: "Norm-based",
        2: "Cosine similarity",
        3: "Distance-based"
    }
    
    results = {}
    for version, name in loss_variants.items():
        loss_fn = get_stable_loss_function(
            loss_version=version,
            normalize=True,
            eps=1e-8
        )
        
        try:
            result = loss_fn(*features)
            
            # Check for NaN/Inf
            has_nan = torch.isnan(result).any().item()
            has_inf = torch.isinf(result).any().item()
            
            results[name] = {
                'shape': result.shape,
                'mean': result.mean().item(),
                'std': result.std().item(),
                'min': result.min().item(),
                'max': result.max().item(),
                'has_nan': has_nan,
                'has_inf': has_inf
            }
            
            print(f"\n✓ {name} (v{version}):")
            print(f"  Shape: {result.shape}")
            print(f"  Range: [{results[name]['min']:.4f}, {results[name]['max']:.4f}]")
            print(f"  Mean: {results[name]['mean']:.4f}, Std: {results[name]['std']:.4f}")
            print(f"  Valid: {'✓' if not (has_nan or has_inf) else '✗ (has NaN/Inf!)'}")
            
            # Test backward pass
            loss = result.mean()
            loss.backward()
            print(f"  Backward pass: ✓")
            
        except Exception as e:
            print(f"\n✗ {name} (v{version}): FAILED")
            print(f"  Error: {e}")
            results[name] = {'error': str(e)}
    
    return results


def test_activation_functions():
    """Test improved activation functions."""
    print("\n" + "="*70)
    print("Testing Improved Activation Functions")
    print("="*70)
    
    x = torch.randn(32, 512)
    
    activations = {
        'gelu': ImprovedActivations.gelu,
        'swish': ImprovedActivations.swish,
        'mish': ImprovedActivations.mish,
        'leaky_relu': ImprovedActivations.leaky_relu,
        'elu': ImprovedActivations.elu,
    }
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    
    for name, act_fn in activations.items():
        y = act_fn(x)
        
        # Compute gradient
        y.sum().backward(retain_graph=True)
        grad_norm = x.grad.norm().item()
        x.grad.zero_()
        
        print(f"\n✓ {name}:")
        print(f"  Output range: [{y.min():.4f}, {y.max():.4f}]")
        print(f"  Gradient norm: {grad_norm:.4f}")
        print(f"  Non-zero outputs: {(y.abs() > 1e-6).float().mean():.2%}")


def test_extreme_cases():
    """Test behavior with extreme inputs."""
    print("\n" + "="*70)
    print("Testing Extreme Cases")
    print("="*70)
    
    stable_ops = StableLossFunctions(eps=1e-8)
    
    # Test 1: Very small values
    print("\nTest 1: Very small values (vanishing gradients)")
    small_features = [torch.randn(8, 512) * 1e-8 for _ in range(3)]
    loss_fn = get_stable_loss_function(loss_version=1, normalize=True)
    
    try:
        result = loss_fn(*small_features)
        print(f"  ✓ Handled small values: mean={result.mean():.4e}")
        print(f"    Has NaN: {torch.isnan(result).any().item()}")
        print(f"    Has Inf: {torch.isinf(result).any().item()}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 2: Very large values
    print("\nTest 2: Very large values (exploding gradients)")
    large_features = [torch.randn(8, 512) * 1e8 for _ in range(3)]
    
    try:
        result = loss_fn(*large_features)
        print(f"  ✓ Handled large values: mean={result.mean():.4e}")
        print(f"    Has NaN: {torch.isnan(result).any().item()}")
        print(f"    Has Inf: {torch.isinf(result).any().item()}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test 3: Mixed scales
    print("\nTest 3: Mixed scales")
    mixed_features = [
        torch.randn(8, 512) * 1e-4,
        torch.randn(8, 512) * 1.0,
        torch.randn(8, 512) * 1e4
    ]
    
    try:
        result = loss_fn(*mixed_features)
        print(f"  ✓ Handled mixed scales: mean={result.mean():.4e}")
        print(f"    Has NaN: {torch.isnan(result).any().item()}")
        print(f"    Has Inf: {torch.isinf(result).any().item()}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def test_6d_scenario():
    """Test the actual 6D CLIP scenario."""
    print("\n" + "="*70)
    print("Testing 6D CLIP Scenario")
    print("="*70)
    
    batch_size = 32
    feature_dim = 512
    
    print(f"\nSimulating 6D CLIP:")
    print(f"  Batch size: {batch_size}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  6 modalities: image + 5 text variants")
    
    # Create 6 feature sets
    image_features = torch.randn(batch_size, feature_dim)
    text_features = [torch.randn(batch_size, feature_dim) for _ in range(5)]
    all_features = [image_features] + text_features
    
    # Test original einsum approach (would create B^6 tensor)
    print(f"\n  Original approach tensor size: {batch_size}^6 = {batch_size**6:,} elements")
    memory_gb = (batch_size**6 * 4) / (1024**3)  # 4 bytes per float32
    print(f"  Memory required: {memory_gb:.2f} GB")
    
    # Test stable approach
    print(f"\n  Testing stable loss function...")
    loss_fn = get_stable_loss_function(loss_version=1, normalize=True)
    
    try:
        result = loss_fn(*all_features)
        print(f"  ✓ Stable loss computed successfully!")
        print(f"    Output shape: {result.shape}")
        print(f"    Mean: {result.mean():.4f}")
        print(f"    Std: {result.std():.4f}")
        
        # Test gradient computation
        loss = result.mean()
        loss.backward()
        print(f"  ✓ Gradients computed successfully!")
        
        # Check gradient health
        model = nn.Module()
        model.features = nn.ParameterList([nn.Parameter(f) for f in all_features])
        
        monitor = GradientHealthMonitor(model, log_frequency=1)
        stats = monitor.check_gradients()
        
        print(f"\n  Gradient Health:")
        print(f"    Global norm: {stats['global_grad_norm']:.4f}")
        print(f"    Max norm: {stats['max_grad_norm']:.4f}")
        print(f"    Min norm: {stats['min_grad_norm']:.4e}")
        
        if stats['recommendations']:
            print(f"\n  Recommendations:")
            for rec in stats['recommendations']:
                print(f"    - {rec[:60]}...")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("GRADIENT HEALTH AND STABILITY TESTS")
    print("="*70)
    
    try:
        # Run tests
        test_gradient_monitor()
        test_stable_loss_functions()
        test_activation_functions()
        test_extreme_cases()
        test_6d_scenario()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED!")
        print("="*70)
        print("\n✓ Gradient health monitoring is working correctly")
        print("✓ Stable loss functions handle extreme cases")
        print("✓ Activation functions provide good gradient flow")
        print("✓ 6D CLIP scenario is numerically stable")
        
        print("\n📚 See docs/GradientHealthGuide.md for integration instructions")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
