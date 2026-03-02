import os
import json
import argparse
import traceback
import datetime
import random
import yaml
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from dataset.data_loaders import get_dataset
from dataset.unlearning_data import (
    compute_step_plan,
    sample_removed_from_remaining,
    build_loaders_by_indices,
)
from utils.fair_unlearning import fair_unlearning
from evaluation.eval import evaluate_unlearning


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return Config(**config_dict)


def save_configs(config, exp_dir: str):
    config_dict = vars(config)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "experiment_configs.json"), 'w') as f:
        json.dump(config_dict, f, indent=4)


def create_model(config, input_tensor, seed: int):
    input_dim = input_tensor.shape[1] if len(input_tensor.shape) == 2 else input_tensor.shape[2]

    if config.model_name == "mlp":
        from model.mlp import MLP
        model = MLP(input_dim=input_dim, hidden=config.mlp.get('hidden_dim', 128), seed=seed)
    elif config.model_name == "lstm":
        from model.lstm import LSTMClassifier
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=config.lstm.get('hidden_dim', 256), num_classes=1, device=config.device, sigmoid=True)
    elif config.model_name == "transformer":
        from model.transformer import build_transformer_from_config
        model = build_transformer_from_config(config, input_dim)
    else:
        raise ValueError(f"Unknown model name: {config.model_name}")

    model.to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    return model

def load_model_weights(model: torch.nn.Module, model_path: str, device: str):
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model

def ensure_step0_origin(config, train_dataset, all_test_datasets, train_data):
    base_dir = os.path.join(config.exp_root.strip(), f"{config.dataset_name}_{config.model_name}_{config.protected_attr}_{config.task_type}")
    step0_dir = os.path.join(base_dir, "step_0")
    os.makedirs(step0_dir, exist_ok=True)

    try:
        origin_model_path = os.path.join(step0_dir, "origin_model.pt")
    except:
        origin_model_path = os.path.join(step0_dir, "model.pth")
    remaining_indices_path = os.path.join(step0_dir, "remaining_indices.npy")

    if not os.path.exists(remaining_indices_path):
        total_samples = len(train_dataset["train_input"]) 
        np.save(remaining_indices_path, np.arange(total_samples, dtype=int))

    if not os.path.exists(origin_model_path):
        from utils.original_model import train_original_model
        print("[step_0] Training original model...")
        set_seed(config.seed)
        save_configs(config, step0_dir)

        protected_attr = config.protected_attr
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                train_dataset["train_input"],
                train_dataset["train_label"].squeeze(),
                train_dataset[f"train_protected_{protected_attr}"]
            ),
            batch_size=config.batch_size,
            shuffle=True
        )

        val_dataset = all_test_datasets[config.dataset_name]

        model = create_model(config, train_data, config.seed)
        loss_fn = nn.BCELoss()
        model, _, _ = train_original_model(model, train_loader, val_dataset, loss_fn, origin_model_path, config)
        print(f"[step_0] Original model saved to {origin_model_path}")
    else:
        print(f"[step_0] Found existing original model at {origin_model_path}")
    
    # Evaluate step 0 if results missing
    results_dir = os.path.join(step0_dir, "results")
    expected_results = os.path.join(results_dir, f"{config.dataset_name}_results_step0.json")
    if not os.path.exists(expected_results):
        try:
            print("[step_0] Evaluating original model (step=0)...")
            if 'model' not in locals():
                model = create_model(config, train_data, config.seed)
                load_model_weights(model, origin_model_path, config.device)
            _ = evaluate_unlearning(
                model=model,
                test_datasets=all_test_datasets,
                exp_dir=step0_dir,
                config=config,
                step=0,
                removed_dataset=(None, None),
            )
        except Exception as e:
            print(f"[step_0] Evaluate failed: {e}")
            traceback.print_exc()

    return step0_dir


def list_parent_dirs_for_step(base_dir: str, step: int) -> List[str]:
    if step == 1:
        return [os.path.join(base_dir, "step_0")]
    prev_step_dir = os.path.join(base_dir, f"step_{step-1}")
    if not os.path.isdir(prev_step_dir):
        return []
    parent_dirs = []
    for name in os.listdir(prev_step_dir):
        path = os.path.join(prev_step_dir, name)
        if os.path.isdir(path) and name.startswith("run_seed_"):
            parent_dirs.append(path)
    parent_dirs.sort()
    return parent_dirs


def _resolve_method_name_from_config_path(config_path: str, fallback: str = "FU") -> str:
    norm = os.path.normpath(config_path)
    parts = norm.split(os.sep)
    if "configs" in parts:
        idx = parts.index("configs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return fallback

def read_removed_indices_for_prev(parent_dir: str, prev_step: int) -> np.ndarray:
    if prev_step <= 0:
        return np.array([], dtype=int)
    path = os.path.join(parent_dir, f"removed_indices_step{prev_step}.npy")
    if not os.path.exists(path):
        return np.array([], dtype=int)
    return np.load(path)

def run_single(config, step: int, run_seed: int, parent_dir: str, train_dataset, all_test_datasets, step_remove_size: int, *, method_key: str):
    base_dir = os.path.join(config.exp_root.strip(), f"{config.dataset_name}_{config.model_name}_{config.protected_attr}_{config.task_type}")
    step_dir = os.path.join(base_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)
    run_dir = os.path.join(step_dir, f"run_seed_{run_seed}")

    if os.path.exists(run_dir):
        print(f"[warn] {run_dir} already exists. Skip this run to avoid overwrite.")
        return

    os.makedirs(run_dir, exist_ok=True)
    save_configs(config, run_dir)

    protected_attr = config.protected_attr
    X_train = train_dataset["train_input"]
    y_train = train_dataset["train_label"].squeeze()
    prot_train = train_dataset[f"train_protected_{protected_attr}"]

    parent_remaining_path = os.path.join(parent_dir, "remaining_indices.npy")
    if not os.path.exists(parent_remaining_path):
        raise FileNotFoundError(f"Parent remaining indices not found: {parent_remaining_path}")
    remaining_indices_prev = np.load(parent_remaining_path)

    set_seed(run_seed)

    removed_indices_t, remaining_indices_t = sample_removed_from_remaining(
        remaining_indices=remaining_indices_prev,
        y_train=y_train,
        protected_attr_data=prot_train,
        step_remove_size=step_remove_size,
        sample_method=getattr(config, 'sample_method', 'random'),
        seed=run_seed,
        target_protected_values=config.target_protected_values,
        label_sampling="stratified",
    )

    removed_loader, remaining_loader = build_loaders_by_indices(
        X=X_train, y=y_train, prot=prot_train,
        removed_indices=removed_indices_t,
        remaining_indices=remaining_indices_t,
        batch_size=config.batch_size,
    )

    model = create_model(config, X_train, run_seed)
    if step == 1:
        parent_model_path = os.path.join(os.path.join(config.exp_root.strip(), f"{config.dataset_name}_{config.model_name}_{config.protected_attr}_{config.task_type}"), "step_0", "origin_model.pt")
    else:
        parent_model_path = os.path.join(parent_dir, "model.pth")
    if not os.path.exists(parent_model_path):
        raise FileNotFoundError(f"Parent model not found: {parent_model_path}")
    load_model_weights(model, parent_model_path, config.device)

    print(f"[step {step}] run_seed={run_seed}: unlearning with method={method_key}...")

    loss_fn = nn.BCELoss()
    stats = fair_unlearning(
        model=model,
        unlearning_loader=removed_loader,
        remaining_loader=remaining_loader,
        loss_fn=loss_fn,
        config=config,
        step=step,
    )

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pth"))
    np.save(os.path.join(run_dir, f"removed_indices_step{step}.npy"), removed_indices_t)
    np.save(os.path.join(run_dir, "remaining_indices.npy"), remaining_indices_t)

    meta = {
        "step": step,
        "seed": run_seed,
        "parent_dir": parent_dir,
        "timestamp": datetime.datetime.now().isoformat(),
        "unlearning_stats": stats
    }
    with open(os.path.join(run_dir, "run_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    X_removed = X_train[removed_indices_t] if len(removed_indices_t) > 0 else None
    y_removed = y_train[removed_indices_t] if len(removed_indices_t) > 0 else None
    removed_tuple = (X_removed, y_removed) if X_removed is not None else (None, None)
    
    results = evaluate_unlearning(
        model=model,
        test_datasets=all_test_datasets,
        exp_dir=run_dir,
        config=config,
        step=step,
        removed_dataset=removed_tuple,
    )

    print(f"[step {step}] run_seed={run_seed}: done. Saved to {run_dir}")

def main(config, *, config_path: str):
    method_name = getattr(config, 'unlearning_method', None).upper()

    if not hasattr(config, 'exp_root') or not getattr(config, 'exp_root'):
        inferred_root = os.path.abspath(os.path.join(os.path.dirname(__file__), f"snapshots_{method_name}"))
        setattr(config, 'exp_root', inferred_root)

    base_dir = os.path.join(config.exp_root.strip(), f"{config.dataset_name}_{config.model_name}_{config.protected_attr}_{config.task_type}")
    os.makedirs(base_dir, exist_ok=True)

    print(f"Config: Dataset={config.dataset_name}, Model={config.model_name}, Device={config.device}")

    dataset_name = config.dataset_name
    data_path = os.path.join(config.data_root.strip(), dataset_name.strip())
    print(f"Loading dataset from {data_path}...")

    if dataset_name == "EICU":
        train_dataset, test_dataset, meta_info = get_dataset(
            dataset_name=dataset_name,
            data_root=data_path,
            device=config.device,
            task_type=config.task_type,
            intersectional=True if config.protected_attr == "ethnicity_age" else False
        )
    elif dataset_name == "MIMIC":
        train_dataset, test_dataset, meta_info = get_dataset(
            dataset_name=dataset_name,
            data_root=data_path,
            device=config.device,
            intersectional=True if config.protected_attr == "ethnicity_age" else False
        )
    elif dataset_name == "YOUR PRIVATE DATASET":
        train_dataset, test_dataset, meta_info = get_dataset(
            dataset_name=dataset_name,
            data_root=data_path,
        )
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    all_test_datasets = {dataset_name: test_dataset}
    if hasattr(config, 'extra_eval_datasets') and config.extra_eval_datasets:
        extra_list = [item.strip() for item in str(config.extra_eval_datasets).split(',') if item and item.strip().lower() != 'none']
        for this_ds in extra_list:
            if this_ds not in all_test_datasets:
                print(f"Loading extra evaluation dataset: {this_ds}...")
                data_path_extra = os.path.join(config.data_root.strip(), this_ds.strip())
                _, this_test_dataset, this_test_meta_info = get_dataset(
                    dataset_name=this_ds,
                    data_root=data_path_extra,
                    device=config.device,
                    normalise_data=False,
                    intersectional=True if config.protected_attr == "ethnicity_age" else False,
                    cal_dist=False
                )
                this_test_meta_info.pop('scaler')
                this_data = meta_info['scaler'].transform(this_test_dataset['test_input'].cpu().detach().numpy())
                this_test_dataset['test_input'] = torch.tensor(this_data).to(config.device).to(torch.float32)
                all_test_datasets[this_ds] = this_test_dataset

    total_samples = len(train_dataset['train_input'])
    step_remove_size, num_steps = compute_step_plan(total_samples, config)
    print(f"Planning: step_remove_size={step_remove_size}, num_steps={num_steps} (+ step_0)")

    ensure_step0_origin(config, train_dataset, all_test_datasets, train_dataset['train_input'])

    if config.target_step == -1:
        steps_to_run = list(range(1, num_steps + 1))
    else:
        if config.target_step == 0:
            steps_to_run = []
        else:
            if config.target_step > num_steps:
                raise ValueError(f"target_step ({config.target_step}) exceeds maximum steps ({num_steps})")
            steps_to_run = [config.target_step]
    print(f"Running steps: {steps_to_run if steps_to_run else '[step_0 only]'}")

    for step in steps_to_run:
        print(f"\n=== Step {step} Start ===")
        parent_dirs = list_parent_dirs_for_step(base_dir, step)
        if not parent_dirs:
            print(f"[warn] Parent directory for Step {step-1} not found. Skip Step {step}")
            continue

        for pidx, parent_dir in enumerate(parent_dirs):
            rng = random.Random(config.seed + step * 1000 + pidx * 100)
            all_seeds: List[int] = []
            while len(all_seeds) < config.num_runs:
                new_seed = rng.randint(0, 1000000)
                if new_seed not in all_seeds:
                    all_seeds.append(new_seed)

            for run_idx, run_seed in enumerate(all_seeds):
                print(f"--- Step {step} | Parent {pidx+1}/{len(parent_dirs)} | Run {run_idx+1}/{config.num_runs} | seed={run_seed}")
                try:
                    run_single(
                        config=config,
                        step=step,
                        run_seed=run_seed,
                        parent_dir=parent_dir,
                        train_dataset=train_dataset,
                        all_test_datasets=all_test_datasets,
                        step_remove_size=step_remove_size,
                        method_key=getattr(config, 'unlearning_method', method_name),
                    )
                except Exception as e:
                    print(f"Error: Step {step}, Parent {pidx+1}, Run {run_idx+1} failed: {e}")
                    traceback.print_exc()
                    continue

        print(f"=== Step {step} Completed ===")

    print("\nAll experiments completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/eICU.yaml')
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config, config_path=args.config_path)
