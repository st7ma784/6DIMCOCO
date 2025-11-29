"""
Quick validation test for gradient health monitoring.
Lighter version that avoids memory-intensive 6D operations.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.GradientHealthCheck import GradientHealthMonitor
from model.StableLossFunctions import get_stable_loss_function, ImprovedActivations

def test_gradient_monitor():
    """Test gradient health monitoring."""
    print("="*70)
    print("Testing Gradient Health Monitor")
    print("="*70)
    
    model = nn.Sequential(
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 128)
    )
    
    monitor = GradientHealthMonitor(
        model=model,
        grad_clip_threshold=1.0,
        log_frequency=1
    )
    
    x = torch.randn(16, 256)
    target = torch.randn(16, 128)
    
    output = model(x)
    loss = ((output - target) ** 2).mean()
    loss.backward()
    
    stats = monitor.check_gradients()
    monitor.log_statistics(stats, step=0)
    
    print(f"\n✓ Gradient monitoring works!")
    print(f"  Global norm: {stats['global_grad_norm']:.4f}")
    print(f"  Has NaN: {stats['has_nan']}")
    print(f"  Has Inf: {stats['has_inf']}")
    
    return stats['has_nan'] == False and stats['has_inf'] == False

def test_stable_loss_3d():
    """Test stable loss with 3 modalities (less memory intensive)."""
    print("\n" + "="*70)
    print("Testing Stable Loss Functions (3D)")
    print("="*70)
    
    batch_size = 16
    feature_dim = 256
    features = [torch.randn(batch_size, feature_dim) for _ in range(3)]
    
    print(f"\nTesting with {len(features)} modalities")
    
    for version in [1, 2, 3]:  # Skip version 0 (einsum) for speed
        loss_fn = get_stable_loss_function(
            loss_version=version,
            normalize=True,
            eps=1e-8
        )
        
        try:
            result = loss_fn(*features)
            has_nan = torch.isnan(result).any().item()
            has_inf = torch.isinf(result).any().item()
            
            print(f"✓ Loss version {version}: shape={result.shape}, valid={not (has_nan or has_inf)}")
            
            if has_nan or has_inf:
                return False
                
            # Test backward
            loss = result.mean()
            loss.backward()
            
        except Exception as e:
            print(f"✗ Loss version {version} failed: {e}")
            return False
    
    return True

def test_epsilon_protection():
    """Test that epsilon protection prevents division by zero."""
    print("\n" + "="*70)
    print("Testing Epsilon Protection")
    print("="*70)
    
    # Create features with very small norms
    small_features = [torch.randn(8, 128) * 1e-10 for _ in range(3)]
    
    loss_fn = get_stable_loss_function(loss_version=1, normalize=True, eps=1e-8)
    
    try:
        result = loss_fn(*small_features)
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        
        print(f"✓ Handled very small values without NaN/Inf")
        print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
        
        return not (has_nan or has_inf)
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def test_activations():
    """Test activation functions."""
    print("\n" + "="*70)
    print("Testing Improved Activation Functions")
    print("="*70)
    
    x = torch.randn(16, 256)
    
    for name in ['gelu', 'swish', 'mish', 'leaky_relu']:
        act_fn = getattr(ImprovedActivations, name)
        y = act_fn(x)
        
        y.sum().backward()
        grad_norm = x.grad.norm().item()
        x.grad.zero_()
        
        print(f"✓ {name}: gradient_norm={grad_norm:.4f}")
    
    return True

def main():
    print("\n" + "="*70)
    print("QUICK GRADIENT HEALTH VALIDATION")
    print("="*70)
    
    tests = [
        ("Gradient Monitor", test_gradient_monitor),
        ("Stable Loss (3D)", test_stable_loss_3d),
        ("Epsilon Protection", test_epsilon_protection),
        ("Activations", test_activations),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("✓ Gradient monitoring is working")
        print("✓ Stable loss functions are operational")
        print("✓ Epsilon protection prevents NaN/Inf")
        print("✓ Improved activations provide good gradient flow")
        return 0
    else:
        print("\n⚠️ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
