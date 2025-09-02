#!/usr/bin/env python3
"""Benchmark script to compare PyTorch vs CuPy loss implementations."""

import torch
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from losses.cupy_losses import CuPyPerformanceProfiler, get_memory_usage_comparison
    CUPY_AVAILABLE = True
except ImportError as e:
    print(f"CuPy not available: {e}")
    CUPY_AVAILABLE = False

def main():
    if not CUPY_AVAILABLE:
        print("Install CuPy to run benchmarks: pip install cupy-cuda11x")
        return
    
    if not torch.cuda.is_available():
        print("CUDA not available - CuPy benefits require GPU")
        return
    
    print("🚀 Benchmarking CuPy vs PyTorch Loss Functions")
    print("=" * 50)
    
    profiler = CuPyPerformanceProfiler()
    
    # Benchmark different loss functions
    loss_functions = ['norm_based', 'einsum']
    batch_sizes = [8, 16, 32]  # Start small to avoid OOM
    
    for loss_name in loss_functions:
        print(f"\n📊 Benchmarking {loss_name} loss:")
        try:
            results = profiler.benchmark_loss_function(
                loss_name=loss_name,
                batch_sizes=batch_sizes,
                feature_dim=256,  # Smaller for initial testing
                num_runs=5
            )
            
            print(f"\n📈 Summary for {loss_name}:")
            for batch_size in batch_sizes:
                speedup = results['speedup'][batch_size]
                print(f"  Batch {batch_size}: {speedup:.2f}x speedup")
                
        except Exception as e:
            print(f"❌ Error benchmarking {loss_name}: {e}")
    
    # Memory usage comparison
    print(f"\n💾 Memory Usage Comparison:")
    try:
        memory_stats = get_memory_usage_comparison()
        print(f"  PyTorch: {memory_stats['pytorch_memory_mb']:.1f} MB")
        print(f"  CuPy: {memory_stats['cupy_memory_mb']:.1f} MB")
        print(f"  Memory Ratio: {memory_stats['memory_ratio']:.2f}x")
    except Exception as e:
        print(f"❌ Error measuring memory: {e}")
    
    print(f"\n✅ Benchmark complete!")

if __name__ == "__main__":
    main()
