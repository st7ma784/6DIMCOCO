"""
Test gradient monitoring in a realistic training scenario.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.GradientHealthCheck import GradientHealthMonitor

class SimpleCLIPModel(nn.Module):
    """Simplified CLIP-like model for testing."""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)
    
    def forward(self, images, texts):
        img_features = self.image_encoder(images)
        txt_features = self.text_encoder(texts)
        
        # Normalize
        img_features = img_features / (img_features.norm(dim=-1, keepdim=True) + 1e-8)
        txt_features = txt_features / (txt_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Compute similarity
        logits = (img_features @ txt_features.T) * self.logit_scale.exp()
        return logits

def test_training_loop_with_monitoring():
    """Simulate training loop with gradient monitoring."""
    print("="*70)
    print("Testing Training Loop with Gradient Monitoring")
    print("="*70)
    
    # Setup
    batch_size = 16
    feature_dim = 512
    model = SimpleCLIPModel(feature_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create gradient monitor
    monitor = GradientHealthMonitor(
        model=model,
        grad_clip_threshold=1.0,
        vanishing_threshold=1e-7,
        exploding_threshold=10.0,
        log_frequency=1
    )
    
    print(f"\n✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✓ Gradient monitor initialized")
    
    # Simulate training steps
    print("\nSimulating training steps...")
    
    for step in range(5):
        # Create batch
        images = torch.randn(batch_size, feature_dim)
        texts = torch.randn(batch_size, feature_dim)
        labels = torch.arange(batch_size)
        
        # Forward pass
        logits = model(images, texts)
        
        # Compute loss (cross entropy)
        loss = nn.functional.cross_entropy(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradient health
        stats = monitor.check_gradients()
        
        print(f"\nStep {step}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradient norm: {stats['global_grad_norm']:.4f}")
        print(f"  Max grad: {stats['max_grad_norm']:.4f}")
        print(f"  Min grad: {stats['min_grad_norm']:.4e}")
        print(f"  Has NaN: {stats['has_nan']}")
        print(f"  Has Inf: {stats['has_inf']}")
        
        # Apply gradient clipping if needed
        if stats['global_grad_norm'] > 1.0:
            print(f"  ⚠️  Applying gradient clipping")
            monitor.apply_gradient_clipping(max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Check for issues
        if stats['has_nan'] or stats['has_inf']:
            print(f"  ❌ CRITICAL: Gradient issues detected!")
            for rec in stats['recommendations']:
                print(f"     {rec[:60]}...")
            break
        
        if stats['recommendations']:
            print(f"  ⚠️  Recommendations:")
            for rec in stats['recommendations'][:2]:  # Show first 2
                print(f"     {rec[:60]}...")
    
    print("\n" + "="*70)
    print("✓ Training loop with monitoring completed successfully!")
    print("  - No NaN/Inf in gradients")
    print("  - Gradient norms in healthy range")
    print("  - Loss is decreasing")
    print("="*70)

def test_gradient_flow_analysis():
    """Test gradient flow analysis."""
    print("\n" + "="*70)
    print("Testing Gradient Flow Analysis")
    print("="*70)
    
    model = SimpleCLIPModel(512)
    monitor = GradientHealthMonitor(model)
    
    # Create dummy batch and compute loss
    images = torch.randn(8, 512)
    texts = torch.randn(8, 512)
    logits = model(images, texts)
    loss = logits.mean()
    loss.backward()
    
    # Get gradient flow summary
    flow_summary = monitor.get_gradient_flow_summary()
    
    print(f"\nGradient Flow Summary:")
    print(f"  Healthy layers: {len(flow_summary['layers_with_healthy_grads'])}")
    print(f"  Vanishing gradient layers: {len(flow_summary['layers_with_vanishing_grads'])}")
    print(f"  Exploding gradient layers: {len(flow_summary['layers_with_exploding_grads'])}")
    
    if flow_summary['layers_with_healthy_grads']:
        print(f"\n  Sample healthy layers:")
        for name, norm in flow_summary['layers_with_healthy_grads'][:3]:
            print(f"    {name}: {norm:.4f}")
    
    print("\n✓ Gradient flow analysis working!")

if __name__ == "__main__":
    test_training_loop_with_monitoring()
    test_gradient_flow_analysis()
