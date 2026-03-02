# Overview
This repository implements **Fair Unlearning (FU)**, a framework for machine unlearning with fairness guarantees. It supports:
- Multiple unlearning strategies with fairness constraints
- Comprehensive evaluation: fairness metrics, membership inference attacks, and model inversion attacks
- Pre-configured support for eICU and MIMIC-IV datasets
- Easy extension to custom private datasets

---

# Quick Start
```bash
# 1. Setup environment
bash build_env.sh  # or: pip install -r requirements.txt

# 2. Prepare data (place under data/<DATASET_NAME>/)
# See "Data Preparation" section for details

# 3. Run with example config
python main.py --config_path configs/MIMIC.yaml

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
└── step_k/run_seed_<seed>/           # Unlearning step k
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

Configuration files are in `configs/` (e.g., `configs/MIMIC.yaml`).

## Key Parameters

### Dataset & Task
```yaml
dataset_name: MIMIC              # MIMIC | EICU | YOUR_PRIVATE_DATASET
data_root: data                  # Data directory root
protected_attr: ethnicity           # ethnicity | ethnicity_age (intersectional)
task_type: mortality             # Task name (EICU: mortality/shock)
extra_eval_datasets: ""          # Optional: comma-separated extra eval datasets
```

### Model Architecture
```yaml
model_name: lstm                 # mlp | lstm | transformer
lstm:                            # Model-specific hyperparameters
  hidden_dim: 128
  num_layers: 2
  # ... (see config files for full options)
```

### Training
```yaml
seed: 42
lr: 0.001
epochs: 100
batch_size: 64
early_stop_patience: 10
```

### Unlearning
```yaml
unlearning_method: FU            # Fair Unlearning
unlearning_lr: 0.0001
unlearning_iterations: 5
weight_cls: 0.1                  # Classification loss weight
weight_fair: 0.5                 # Fairness loss weight
weight_remain: 0.3               # Remaining data utility weight
```

### Forgetting Schedule
```yaml
remove_ratio: 0.1                # Max forgetting ratio (e.g., 10%)
step_ratio: 0.01                 # Increment per step (e.g., 1%)
target_step: -1                  # -1: all steps | 0: only eval original | >0: specific step
num_runs: 3                      # Repeated runs per step
sample_method: random            # Sampling strategy (combine with target_protected_values for targeted)
```

**Example**: `remove_ratio=0.1` + `step_ratio=0.01` → 10 forgetting steps (1%, 2%, ..., 10%) + original model (step_0)

### Evaluation
```yaml
eval_fairness: 1                 # 0: off | 1: on
eval_mia: 1                      # Membership Inference Attack
eval_model_inversion: 1          # Model Inversion Attack
mia_kwargs:                      # MIA-specific hyperparameters
  # ... (see config for details)
model_inversion_kwargs:          # Model inversion-specific hyperparameters
  # ... (see config for details)
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
   - Results saved under `step_k/run_seed_xxx/results/`

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
