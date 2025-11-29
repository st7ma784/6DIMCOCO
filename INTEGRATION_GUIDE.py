"""
Quick Integration Example: Add Gradient Monitoring to Existing Training

This shows the minimal changes needed to add gradient monitoring 
to your existing trainclip_v534DIM.py or similar files.

OPTION 1: Minimal Integration (Just add these lines)
=================================================== 

In your LightningCLIPModule class:

1. Add import at top of file:
   from model.GradientHealthCheck import GradientHealthMonitor

2. In __init__, add this after super().__init__():
   
   self.grad_monitor = GradientHealthMonitor(
       model=self,
       grad_clip_threshold=1.0,
       log_frequency=100  # Log every 100 steps
   )

3. Add this method to your class:

   def on_after_backward(self):
       '''Called after backward pass'''
       if self.global_step % 100 == 0:  # Check every 100 steps
           stats = self.grad_monitor.check_gradients()
           
           # Log to wandb if available
           if hasattr(self.logger, 'experiment'):
               self.logger.experiment.log({
                   'grad/global_norm': stats['global_grad_norm'],
                   'grad/has_nan': int(stats['has_nan']),
                   'grad/has_inf': int(stats['has_inf']),
               }, step=self.global_step)
           
           # Apply clipping if needed
           if stats['global_grad_norm'] > 10.0:
               self.grad_monitor.apply_gradient_clipping(max_norm=1.0)

4. In configure_optimizers, add gradient clipping:

   return {
       'optimizer': optimizer,
       'lr_scheduler': lr_scheduler,
       'gradient_clip_val': 1.0,  # <-- Add this line
       'gradient_clip_algorithm': 'norm'  # <-- Add this line
   }

That's it! You now have:
- Real-time gradient monitoring
- Automatic NaN/Inf detection  
- Gradient clipping
- WandB logging


OPTION 2: Full Integration (Use the mixin)
=========================================

If you want all features with minimal code:

1. Add imports:
   from model.IntegrateGradientHealthChecks import EnhancedTrainingMixin

2. Change class definition:
   # Before:
   class LightningCLIPModule(base):
   
   # After:
   class LightningCLIPModule(EnhancedTrainingMixin, base):

3. In __init__, add one line:
   self.setup_gradient_monitoring(enable_gradient_monitoring=True)

4. Add this method:
   def on_after_backward(self):
       self.check_and_log_gradients(
           step=self.global_step,
           apply_clipping=True,
           max_grad_norm=1.0
       )


WHAT YOU'LL SEE IN LOGS
=======================

Console output every 100 steps:

  Step 100 - Gradient Health: Global Norm=11.14, Max=5.84, Min=6.24e-02
  ⚠️ Exploding gradients detected (norm=11.14)!
     Consider: (1) Gradient clipping, (2) Reducing learning rate

WandB metrics:
  - grad/global_norm
  - grad/max_norm
  - grad/min_norm
  - grad/has_nan
  - grad/has_inf


RECOMMENDED HYPERPARAMETER CHANGES
==================================

Based on the 6D loss complexity:

1. Lower learning rate:
   learning_rate = 1e-4  # Instead of 2e-3

2. Enable gradient clipping:
   gradient_clip_val = 1.0

3. Consider using stable loss:
   from model.StableLossFunctions import get_stable_loss_function
   self.calculate_loss = get_stable_loss_function(
       loss_version=1,  # Norm-based, more stable
       normalize=True,
       eps=1e-8
   )

4. For MarianMTModel config:
   activation_function="gelu"  # Instead of "swish"


DEBUGGING WORKFLOW
==================

When you see issues in training:

1. Check console for gradient health messages
2. Look at WandB grad/* metrics
3. Follow the automatic recommendations
4. Adjust learning rate or clipping based on feedback

Example:
- If "exploding gradients": Reduce LR or increase clipping
- If "vanishing gradients": Check activation functions
- If NaN/Inf: Check loss function, add more epsilon protection


PERFORMANCE IMPACT
==================

Minimal:
- ~1-2% slower training (logging overhead)
- Can disable after debugging by setting log_frequency=1000
- No impact when monitoring is disabled


FILES TO REFERENCE
==================

- Full example: model/example_enhanced_training.py
- Documentation: docs/GradientHealthGuide.md  
- Quick ref: model/README_GRADIENT_HEALTH.md
- Tests: model/test_monitoring_integration.py

"""

print(__doc__)

# Example of what your training_step might look like:
print("\n" + "="*70)
print("EXAMPLE: Enhanced training_step")
print("="*70)
print("""
def training_step(self, batch, batch_idx):
    im, captions = batch[0], batch[1]
    
    # Forward pass
    logits = self(im, *[captions[:, i] for i in range(captions.shape[1])])
    logits = logits * self.logit_scale.exp()
    
    # Check logits health (optional but helpful)
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print(f"⚠️ NaN/Inf in logits at step {self.global_step}")
    
    # Compute loss
    losses = [self.loss(logits.permute(*i), labels, alpha=self.alpha) 
              for i in permutes]
    loss = self.meanloss(I=[losses[0]], T=losses[1:]).mean()
    
    # Logging
    self.log('train_loss', loss, prog_bar=True)
    self.log('logit_scale', self.logit_scale.exp())
    
    return {"loss": loss}

def on_after_backward(self):
    '''Gradient checking happens here automatically'''
    if self.global_step % 100 == 0:
        stats = self.grad_monitor.check_gradients()
        
        # Take action if needed
        if stats['has_nan'] or stats['has_inf']:
            print(f"🚨 CRITICAL at step {self.global_step}")
            print("Consider reducing learning rate immediately")
        
        # Adaptive clipping
        if stats['global_grad_norm'] > 10.0:
            self.grad_monitor.apply_gradient_clipping(max_norm=1.0)
""")
