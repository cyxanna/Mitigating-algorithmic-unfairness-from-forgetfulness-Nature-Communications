# Overview
This repository implements **Fair Unlearning (FU)**, a framework for machine unlearning with fairness guarantees. 

---

# Quick Start
```bash
# 1. Setup environment
bash build_env.sh  # or: pip install -r requirements.txt

# 2. Prepare data (place under data/<DATASET_NAME>/)
# See "Data Preparation" section for details

# 3. Run with example config
python main.py --config_path configs/example.yaml

# 4. Check results
# Output: snapshots_FU/<dataset>_<model>_<protected_attr>_<task_type>/
```

**Recommended Environment**: Ubuntu 22.04.5 LTS, Python 3.12, CUDA 12.1, PyTorch 2.4.1 (We do not test under Windows environment.)

---

# Project Structure

## Code Organization
```
├── configs/              # YAML configuration files
├── data/                 # Dataset root (user-provided)
├── dataset/              # Data loading & Unlearning data preparation
├── model/                # Model architectures (MLP/LSTM/Transformer)
├── evaluation/           # Evaluation modules (fairness, MIA, model inversion)
├── utils/                # Original model training & fair unlearning utilities
├── snapshots_FU/         # Default output directory (customizable)
└── main.py               # Main entry point
```

## Output Structure
```
snapshots_FU/<dataset>_<model>_<protected_attr>_<task_type>/
├── step_0/                           # Original model (before unlearning)
│   ├── origin_model.pt
│   ├── remaining_indices.npy
│   └── results/<dataset>_results_step0.json
└── step_k/seed_<unlearning_seed>/    # Unlearning step k
    ├── model.pth
    ├── removed_indices_stepk.npy
    ├── remaining_indices.npy
    ├── run_meta.json                 # Hyperparams, timestamp, stats
    └── results/                      # Evaluation outputs
```

---

# Data Preparation

## Supported Datasets

| Dataset | Access | Required Files |
|---------|--------|----------------|
| **eICU** | Public: https://www.physionet.org/content/eicu-crd/2.0/ | `mortality.h5`, `shock.h5` |
| **MIMIC-IV** | Public: https://physionet.org/content/mimiciv/2.0/ | `dataset_X.npz`, `dataset_Y.npz`, `dataset_metadata.json` |
| **CURIAL (OUH/UHB/PUH/BH)** | Request per institutional policy | Institution-specific |

## Preprocessing

### MIMIC-IV
Follow the [MIMIC-IV-Data-Pipeline](https://github.com/healthylaife/MIMIC-IV-Data-Pipeline) to generate:
- `dataset_X.npz`: feature matrix
- `dataset_Y.npz`: labels
- `dataset_metadata.json`: metadata (protected attributes, etc.)

### eICU
We follow the preprocessing pipeline from [eICU Benchmark](https://github.com/mostafaalishahi/eICU_Benchmark) to build cohorts for the **mortality** and **shock**(phenotype) prediction tasks.  
After running that pipeline, you need to convert the processed data into a single HDF5 file per task and place it under your `data_root`:
- `mortality.h5` for mortality prediction
- `shock.h5` for the shock prediction task

Each HDF5 file is expected to contain at least the following datasets (keys):
- `X_train`  : training features, shape `(N_train, T, D)`
- `X_test`   : test features, shape `(N_test, T, D)`
- `y_train`  : training labels
- `y_test`   : test labels
- `train_gender`       : protected attribute (gender) for the training set
- `train_ethnicity`    : protected attribute (ethnicity) for the training set
- `train_age_group`    : protected attribute (age group) for the training set
- `test_gender`        : protected attribute (gender) for the test set
- `test_ethnicity`     : protected attribute (ethnicity) for the test set
- `test_age_group`     : protected attribute (age group) for the test set

All subsequent processing steps (one-hot encoding of categorical variables, concatenation of features, and etc.) are implemented in `dataset/data_loaders.py` via the `get_EICU_dataset` function. 


## Expected Data Layout
```
data/
├── EICU/
│   ├── mortality.h5
│   └── shock.h5
├── MIMIC/
│   ├── dataset_metadata.json
│   ├── dataset_X.npz
│   └── dataset_Y.npz
└── YOUR_PRIVATE_DATASET/
    └── <your data files>
```

---

# Configuration Guide

Configuration files are in `configs/` (`template.yaml` is the template; `example.yaml` is a runnable example).

**Note**: > **Note**: The values shown below are examples/defaults and should be adjusted based on your dataset. This repository also provides additional configuration options (e.g., gradient-orthogonalization method, fairness-loss variant) to facilitate experimentation, along with a learning-rate schedule that improves stability for cumulative unlearning.

## Complete Parameter Reference

### Dataset & Task
```yaml
dataset_name: MIMIC              # MIMIC | EICU | YOUR_PRIVATE_DATASET
data_root: data                  # Data directory root
protected_attr: ethnicity        # ethnicity | gender
task_type: mortality             # Task name (EICU: mortality/shock)
extra_eval_datasets: ""          # Optional: comma-separated extra eval datasets
```

### Model Architecture
```yaml
model_name: lstm                 # mlp | lstm | transformer
device: cuda                     # cuda | cuda:0 | cuda:1 | cpu

# MLP-specific (when model_name: mlp)
mlp:
  hidden_dim: 128

# LSTM-specific (when model_name: lstm)
lstm:
  hidden_dim: 256

# Transformer-specific (when model_name: transformer)
transformer:
  dim: 64                        # Model dimension
  depth: 2                       # Number of transformer layers
  heads: 8                       # Number of attention heads
  ff_dim: 256                    # Feed-forward dimension
  dropout: 0.1                   # Dropout rate
```

### Training (Original Model)
```yaml
seed: 42                         # Random seed for training
val_split_seed: 42               # Seed for train/val split
lr: 0.001                        # Learning rate
epochs: 100                      # Maximum training epochs
batch_size: 64                   # Batch size
early_stop_patience: 10          # Early stopping patience
```

### Unlearning Method
```yaml
unlearning_method: FU            # Fair Unlearning method

# FU loss weights
unlearning_lr: 0.0001            # Unlearning learning rate
unlearning_epochs: 5             # Gradient update epochs per step
weight_cls: 0.1                  # Forget gradient weight (unlearning strength)
weight_fair: 0.5                 # Fairness gradient weight
weight_remain: 0.3               # Remaining data utility weight
```

### FU Implementation Variants

These parameters control the algorithmic variants of Fair Unlearning. Each option has distinct characteristics suited for different scenarios.

```yaml
fairness_loss_type: pairwise_squared_diff
gram_schmidt_type: orthonormal_basis
normalize_forget_grad: false
```

**fairness_loss_type** - Fairness loss computation method:
| Option | Characteristics | Use Case |
|--------|-----------------|----------|
| `pairwise_squared_diff` | Computes squared differences across all group pairs. Complexity grows quadratically with number of groups. | Fewer groups; strict balancing across all group pairs required. |
| `group_mean_diff` | Computes deviation of each group from the global mean. Complexity grows linearly. | More groups; focus on deviation from overall average. |

**gram_schmidt_type** - Gradient orthogonalization method:
| Option | Characteristics | Use Case |
|--------|-----------------|----------|
| `orthonormal_basis` | Constructs a complete orthonormal basis, ensuring all gradient components are fully orthogonal. More computationally complete. | Strict gradient orthogonality guarantees required. |
| `sequential_projection` | Projects gradients sequentially in order. More computationally lightweight. | High-dimensional gradients; relaxed orthogonality requirements. |

**normalize_forget_grad** - Whether to normalize the forget gradient:
| Option | Characteristics | Use Case |
|--------|-----------------|----------|
| `true` | Normalizes to unit length, removing magnitude influence and preserving only directional information. | Gradient magnitudes vary significantly across loss terms. |
| `false` | Preserves original gradient magnitude. | Gradient magnitude carries meaningful information. |

### Forgetting Schedule
```yaml
exp_root: ""                     # Output directory (default: snapshots_FU/)
unlearning_seed: 100             # Seed for sampling removed data and randomness in unlearning
target_step: -1                  # Which steps to run
                                 #   - -1: run all steps
                                 #   - 0: only train/eval original model (step_0)
                                 #   - >0: run specific step only
sample_method: random            # How to sample data for removal
                                 #   - "random": uniform random sampling
                                 #   - "majority": sample from majority group
remove_ratio: 0.1                # Total removal ratio (e.g., 0.1 = 10% of training data)
step_ratio: 0.01                 # Removal ratio per step (e.g., 0.01 = 1% per step)
target_protected_values: null    # Target specific protected attribute values for removal (null = all)
```

**Example**: `remove_ratio=0.1` + `step_ratio=0.01` → 10 forgetting steps (1%, 2%, ..., 10%) + original model (step_0)

### Evaluation
```yaml
eval_fairness: 1                 # Fairness metrics (0: off, 1: on)
eval_mia: 1                      # Membership Inference Attack (0: off, 1: on)
eval_model_inversion: 1          # Model Inversion Attack (0: off, 1: on)
```

### MIA Hyperparameters
```yaml
mia_kwargs:
  seed: 1234                     # Random seed for MIA evaluation
  sample_test_size: 500          # Number of test samples for MIA (null = same size for test data)
```

### Model Inversion Hyperparameters
```yaml
model_inversion_kwargs:
  seed: 1234                     # Random seed for model inversion
  iters: 100                     # Optimization iterations for reconstruction
  noise_std_range: [0.01, 0.1]   # Range for initial noise standard deviation
  l1_reg: 0.001                  # L1 regularization weight
  k: 10                          # k for KNN distance computation
  multi_restart: 1               # Number of random restarts
  inv_per_class: 256             # Number of inversions per class
```

---

# Pipeline Workflow

The `main.py` script orchestrates the following workflow:

1. **Initialization**
   - Load configuration from YAML
   - Set output directory: `snapshots_FU/<dataset>_<model>_<protected_attr>_<task_type>/`

2. **Data Loading**
   - Load dataset via `get_dataset` in `dataset/data_loaders.py`
   - Split train/val/test sets with protected attributes

3. **Forgetting Schedule**
   - Compute forgetting steps using `compute_step_plan` (based on `remove_ratio` / `step_ratio`)

4. **Step 0: Original Model**
   - Train original model and save to `step_0/origin_model.pt`
   - Reuse existing model if available
   - Recompute missing evaluation results

5. **Step k: Unlearning (k > 0)**
   - Sample removal indices from remaining training set
   - Apply Fair Unlearning (FU) algorithm
   - Save unlearned model, indices, and metadata

6. **Evaluation**
   - **Fairness**: demographic parity, equalized odds, etc.
   - **MIA**: membership inference attack (AUROC)
   - **Model Inversion**: model inversion attack (KNN Distance)
   - Results saved under `step_k/seed_<unlearning_seed>/results/`

---

# Baselines

Most of the baseline methods compared in our experiments provide open-source implementations that can be used directly:

| Method | Repository |
|--------|------------|
| **CR** (Certified Removal) | [facebookresearch/certified-removal](https://github.com/facebookresearch/certified-removal) |
| **CF** (Catastrophic Forgetting) | [shash42/Evaluating-Inexact-Unlearning](https://github.com/shash42/Evaluating-Inexact-Unlearning) |
| **SCRUB** | [meghdadk/SCRUB](https://github.com/meghdadk/SCRUB) |

---

# Extending to Private Datasets

## Step-by-Step Guide

1. **Prepare Data**
   ```bash
   mkdir -p data/YOUR_DATASET_NAME
   # Place your data files here
   ```

2. **Implement Data Loader**
   Edit `dataset/data_loaders.py`:
   ```python
   def get_your_private_dataset(config):
       # Load your data
       # Return: X_train, y_train, X_val, y_val, X_test, y_test, protected_attrs
       pass
   
   # Register in DATASET_FUNCS
   DATASET_FUNCS = {
       'MIMIC': get_mimic_dataset,
       'EICU': get_eicu_dataset,
       'YOUR_DATASET_NAME': get_your_private_dataset,
   }
   ```

3. **Create Configuration**
   Copy and modify an existing config:
   ```yaml
   dataset_name: YOUR_DATASET_NAME
   protected_attr: your_protected_attribute
   task_type: your_task_name
   # ... other settings
   ```

4. **Run**
   ```bash
   python main.py --config_path configs/YOUR_DATASET_NAME.yaml
   ```

---

For questions or issues, please refer to the code documentation or open an issue.
