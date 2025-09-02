#!/usr/bin/env python3
"""
Entry point script for training that handles path setup properly.
This replaces direct calls to launch.py and ensures imports work correctly.
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.launch import train, wandbtrain

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CLIP training")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type")
    
    args = parser.parse_args()
    
    # Default config
    config = {
        "batch_size": 16,
        "learning_rate": 2e-3,
        "precision": 16,
        "embed_dim": 512,
        "codeversion": 4,
        "transformer_width": 512,
        "transformer_heads": 32,
        "transformer_layers": 4,
        "JSE": False,
    }
    
    if args.wandb:
        wandbtrain(config=config, devices=args.devices, accelerator=args.accelerator)
    else:
        train(config=config, devices=args.devices, accelerator=args.accelerator)
