# Project Structure Documentation

## Directory Organization

The project has been reorganized into a clean, modular structure:

### 📁 Core Directories

- **`src/`** - Core source code with modular architecture
  - `config/` - Type-safe configuration management
  - `losses/` - Refactored loss function implementations
  
- **`model/`** - CLIP model implementations (various dimensions)
  - `trainclip_v53.py` - 6D CLIP model
  - `trainclip_v534DIM.py` - 4D variant
  - `trainclip_v533DIM.py` - 3D variant
  - `LossCalculation.py` - Legacy loss functions (being phased out)

- **`scripts/`** - Training and analysis scripts
  - `run_training.py` - Main entry point for training
  - `launch.py` - Training orchestration
  - `CKA_*.py` - CKA analysis scripts
  - `benchmark_cupy.py` - Performance benchmarking
  - `trainagent.py` - Slurm job management
  - `captum*.py` - Model interpretability

- **`data_builders/`** - Dataset construction scripts
  - `BuildCNDataset.py` - Chinese dataset builder
  - `BuildImagenet.py` - ImageNet dataset builder
  - `BuildLAION.py` - LAION dataset builder
  - `BuildSpainDataSet.py` - Spanish dataset builder
  - Various evaluation dataset builders

- **`notebooks/`** - Jupyter notebooks for analysis
  - `ContrastiveDemo.ipynb` - Contrastive learning demos
  - `plotresults.ipynb` - Results visualization
  - `wandbAnalysis.ipynb` - Weights & Biases analysis
  - `runonAzure.ipynb` - Azure deployment notebook

- **`results/`** - Training outputs and visualizations
  - PNG plots and result images
  - PKL result files
  - JSON trace files

- **`experiments/`** - Experimental configurations and temporary files

### 🔧 Configuration Files

- **`tests/`** - Comprehensive test suite
- **`docs/`** - Sphinx documentation
- **`Visualisations/`** - Web-based result visualization
- **`dependencies/`** - Conda environment files

## Import Path Changes

### Before Reorganization
```python
# Old structure - files in root
from model.LossCalculation import *
import launch
```

### After Reorganization
```python
# New structure - organized directories
from model.LossCalculation import *
from scripts.launch import train, wandbtrain
```

### Entry Points

- **Training**: `python scripts/run_training.py`
- **Dataset Building**: `python data_builders/Build*.py`
- **Analysis**: `python scripts/CKA_*.py`
- **Benchmarking**: `python scripts/benchmark_cupy.py`

## Docker Updates

The Dockerfile has been updated to use the new entry point:
```dockerfile
CMD python scripts/run_training.py --dir /data --wandb
```

## Benefits of New Structure

1. **Clean Root Directory** - No more scattered files
2. **Logical Grouping** - Related functionality grouped together
3. **Better Imports** - Clear import paths with proper path handling
4. **Maintainability** - Easier to navigate and understand
5. **Scalability** - Easy to add new components in appropriate directories
