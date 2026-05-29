import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import json
from evaluation.utils import cutoff_youdens_j, compute_auroc, fairness_funcs

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from evaluation.utils import decomposed_equalized_odds_difference, compute_dp_sd


def convert_to_json_serializable(obj):

    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


METRICS_AU = {
    'au-roc': compute_auroc,
}


def compute_fairness_metrics(y_true, y_pred_class, y_protected):
    this_fairness = dict()
    
    # Decomposed EOD metrics
    decomposed_eod = decomposed_equalized_odds_difference(
        y_true=y_true, y_pred=y_pred_class, sensitive_features=y_protected)

    this_fairness['decomposed_eod'] = decomposed_eod
    this_fairness['tpr_sd'] = decomposed_eod['tpr_sd']
    this_fairness['fpr_sd'] = decomposed_eod['fpr_sd']

    # Demographic Parity SD
    demographic_parity_sd = compute_dp_sd(y_pred_class, y_protected, return_pairwise=True)
    this_fairness['demographic_parity_sd'] = demographic_parity_sd
    
    return this_fairness


def evaluate_prob_preds(
        y_true,
        y_pred_prob,
        y_protected_dict=None,
        best_threshold_fn=cutoff_youdens_j,
        threshold_override=None):
    """
    Evaluate the probability predictions
    Args:
        y_true: an 1D array-like of true labels
        y_pred_prob: an 1D array-like of probability predictions
        y_protected_dict: a dictionary of sensitive information for fairness evaluation;
            if None, will not evaluate fairness
            each key should be the sensitive attribute's name, e.g. 'gender', 'ethnicity' etc.
            each value should be an 1D array-like labels of sensitive attributes
        threshold_override: if provided, use this threshold instead of computing from y_true/y_pred_prob
    Returns:
        A dict of evaluation results
    """
    y_pred_prob, y_true = np.array(y_pred_prob).flatten(), np.array(y_true).flatten()
    if threshold_override is not None:
        best_threshold = threshold_override
    else:
        best_threshold = best_threshold_fn(true=y_true, pred=y_pred_prob)
    y_pred_class = (y_pred_prob >= best_threshold).astype(int)


    eval_res = dict()

    for metric_name, metric_func in METRICS_AU.items():
        eval_res[metric_name] = metric_func(true=y_true, pred=y_pred_prob)

    if y_protected_dict is not None:
        for name, y_protected in y_protected_dict.items():
            y_protected = np.array(y_protected).flatten()
            eval_res[f'fairness_{name}'] = compute_fairness_metrics(
                y_true=y_true, y_pred_class=y_pred_class, y_protected=y_protected,)

    return eval_res, best_threshold



def _forward_collect(model, dataset, device):
    """Forward pass through a dataset and collect predicted probabilities and loss."""
    tds = TensorDataset(dataset["test_input"], dataset["test_label"].squeeze())
    loader = DataLoader(tds, batch_size=512, shuffle=False)
    loss_fn = torch.nn.BCELoss()

    model.eval()
    y_preds_prob = []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, dict):
                outputs = outputs['preds']
            total_loss += loss_fn(outputs[:, 0], labels).item()
            y_preds_prob.extend(outputs.cpu().numpy())

    total_loss /= len(loader)
    y_preds_prob = np.asarray(y_preds_prob).flatten()
    return y_preds_prob, total_loss


def evaluate_cls_fairness(model, test_dataset, device, eval_fairness, protected_attr, val_dataset=None):

    # Forward pass on test set
    y_preds_prob, val_loss = _forward_collect(model, test_dataset, device)

    # Determine threshold: use val_dataset if provided, otherwise fall back to test set
    if val_dataset is not None:
        val_preds_prob, _ = _forward_collect(model, val_dataset, device)
        val_labels = val_dataset["test_label"].detach().cpu().numpy().flatten()
        best_threshold = cutoff_youdens_j(true=val_labels, pred=val_preds_prob)
    else:
        best_threshold = None  # will be computed on test set inside evaluate_prob_preds

    labels = test_dataset["test_label"].detach().cpu().numpy()

    if eval_fairness:
        protected_attr_dict = {
            protected_attr: test_dataset[f"test_protected_{protected_attr}"].cpu().numpy()
        }
    else:
        protected_attr_dict = None

    eval_res, best_threshold = evaluate_prob_preds(
        y_true=labels,
        y_pred_prob=y_preds_prob,
        y_protected_dict=protected_attr_dict,
        threshold_override=best_threshold,
    )
    eval_res['best_threshold'] = float(best_threshold)

    return eval_res, val_loss, y_preds_prob


def evaluate_mia(model, X_removed, X_test, y_removed, y_test, device, seed=1234, sample_test_size=None):
    """
    Evaluate membership inference. Default is a training-based attack classifier.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.eval()

    def _forward_prob(batch_x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            outputs = model(batch_x.to(device))
            if isinstance(outputs, dict):
                if 'preds' in outputs:
                    outputs = outputs['preds']
                elif 'logits' in outputs:
                    outputs = torch.sigmoid(outputs['logits'])
                else:
                    for v in outputs.values():
                        outputs = v
                        break

            probs = outputs
            if probs.dtype != torch.float32 and probs.dtype != torch.float64:
                probs = probs.float()

            probs = torch.clamp(probs, 1e-7, 1. - 1e-7)
            if probs.ndim > 1:
                probs = probs[:, 0]
            return probs.detach().cpu().numpy().reshape(-1, 1)


    def _build_features(y_true_np: np.ndarray, prob_np: np.ndarray) -> np.ndarray:

        p = prob_np.reshape(-1)
        eps = 1e-7
        p = np.clip(p, eps, 1.0 - eps)
        y = y_true_np.reshape(-1).astype(np.float32)

        max_prob = np.maximum(p, 1.0 - p)
        margin = np.abs(p - 0.5)
        entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        logit = np.log(p / (1.0 - p))
        feats = np.stack([max_prob, margin, entropy, loss, logit], axis=1).astype(np.float32)
        return feats

    # samples same number of samples from test set
    test_size = X_test.shape[0]
    if sample_test_size is not None:
        test_indices = np.random.choice(np.arange(test_size), size=sample_test_size, replace=False)
    else:
        test_indices = np.random.choice(np.arange(test_size), size=X_test.shape[0], replace=False)
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]
    
    # Training-based attack
    m_prob = _forward_prob(X_removed)
    n_prob = _forward_prob(X_test)
    y_removed_np = y_removed.detach().cpu().numpy().reshape(-1)
    y_test_np = y_test.detach().cpu().numpy().reshape(-1)

    X_m = _build_features(y_removed_np, m_prob)
    X_n = _build_features(y_test_np, n_prob)
    X_all = np.concatenate([X_m, X_n], axis=0)
    y_all = np.concatenate([np.ones(X_m.shape[0], dtype=np.float32), np.zeros(X_n.shape[0], dtype=np.float32)], axis=0)

    # Train/val split
    rng = np.random.RandomState(seed)
    indices = np.arange(X_all.shape[0])
    rng.shuffle(indices)
    split = int(0.7 * X_all.shape[0])
    idx_tr, idx_va = indices[:split], indices[split:]
    X_tr, y_tr = X_all[idx_tr], y_all[idx_tr]
    X_va, y_va = X_all[idx_va], y_all[idx_va]

    # Logistic Regression attack (scikit-learn)
    attack = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
    )
    attack.fit(X_tr, y_tr)
    scores_va = attack.predict_proba(X_va)[:, 1]
    mia_auc = roc_auc_score(y_va, scores_va)
    return float(mia_auc)


def inversion_knn_distance_by_label(
    model,
    X_removed: torch.Tensor,
    y_removed: torch.Tensor,
    device,
    label_counts: dict,
    iters: int = 100,
    learning_rate: float = 0.001,
    noise_std_range: tuple = (0.01, 0.1),
    l1_reg: float = 1e-3,
    k: int = 1,
    multi_restart: int = 1,
    seed: int = 1234,
    inv_per_class: int = None):

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.eval()
    
    is_timeseries = (X_removed.ndim == 3)
    if is_timeseries:
        seq_len = int(X_removed.shape[1])
        feature_dim = int(X_removed.shape[2])
        flat_dim = seq_len * feature_dim
    else:
        feature_dim = int(X_removed.shape[1])
        flat_dim = feature_dim

    labels = []
    for lab, n in label_counts.items():
        n_use = int(max(0, n))
        if n_use > 0:
            labels.append(np.full((n_use,), int(lab), dtype=np.int64))
    if not labels:
        return {"per_class": {}, "overall": None}
    y_targets = np.concatenate(labels, axis=0)

    noise_std_range = tuple(map(float, noise_std_range))
    noise_std = rng.uniform(noise_std_range[0], noise_std_range[1])

    def reconstruct_all(y_t: np.ndarray) -> np.ndarray:
        N = int(y_t.shape[0])
        re_best = None
        best = None
        bs = 1024
        for r in range(max(1, int(multi_restart))):
            parts = []
            for s in range(0, N, bs):
                e = min(N, s + bs)
                bsz = e - s
                y_chunk = torch.tensor(y_t[s:e], device=device, dtype=torch.float32)
                if is_timeseries:
                    x_hat = torch.zeros((bsz, seq_len, feature_dim), device=device, requires_grad=True)
                else:
                    x_hat = torch.zeros((bsz, feature_dim), device=device, requires_grad=True)
                with torch.no_grad():
                    x_hat.add_(torch.randn_like(x_hat) * float(noise_std))
                opt = torch.optim.Adam([x_hat], lr=float(learning_rate))
                for _ in range(int(iters)):
                    opt.zero_grad()
                    out = model(x_hat)
                    if isinstance(out, dict):
                        preds = out.get('preds', None)
                        if preds is None and 'logits' in out:
                            preds = torch.sigmoid(out['logits'])
                        if preds is None:
                            for v in out.values():
                                preds = v
                                break
                    else:
                        preds = out
                    if preds.ndim > 1:
                        preds = preds[:, 0]
                    loss = torch.nn.functional.binary_cross_entropy(preds, y_chunk) + float(l1_reg) * torch.sum(torch.abs(x_hat))
                    loss.backward()
                    opt.step()
                parts.append(x_hat.detach().cpu().view(bsz, -1).numpy())
            re_np = np.vstack(parts) if parts else np.zeros((0, flat_dim), dtype=np.float32)

            with torch.no_grad():
                rt = torch.tensor(re_np, dtype=torch.float32)
                if is_timeseries:
                    rt = rt.view(-1, seq_len, feature_dim)
                out = model(rt.to(device))
                if isinstance(out, dict):
                    p = out.get('preds', None)
                    if p is None and 'logits' in out:
                        p = torch.sigmoid(out['logits'])
                else:
                    p = out
                if p.ndim > 1:
                    p = p[:, 0]
                p = p.detach().cpu().numpy().reshape(-1)
                y_idx = y_t.astype(int).clip(0, 1)
                sel = np.where(y_idx == 1, p, 1.0 - p)
            if best is None:
                best = sel
                re_best = re_np
            else:
                pick = sel > best
                best[pick] = sel[pick]
                re_best[pick] = re_np[pick]
        return re_best

    Xhat = reconstruct_all(y_targets)

    def to_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy().reshape(x.shape[0], -1).astype(np.float32)

    Z_for = to_np(X_removed)
    y_for = y_removed.detach().cpu().numpy().reshape(-1)
    Z_hat = Xhat.astype(np.float32)

    def knn_mean(query: np.ndarray, gallery: np.ndarray, topk: int) -> np.ndarray:
        Qa2 = np.sum(query * query, axis=1, keepdims=True)
        Gb2 = np.sum(gallery * gallery, axis=1, keepdims=True).T
        cross = query @ gallery.T
        dist2 = np.maximum(Qa2 + Gb2 - 2.0 * cross, 0.0)
        k_eff = min(int(topk), gallery.shape[0])
        part = np.partition(dist2, kth=k_eff-1, axis=1)[:, :k_eff]
        dis = np.sqrt(np.mean(part, axis=1)).astype(np.float32)
        return dis

    per_class = {}
    all_d_rem = []
    all_d_for = []
    offset = 0
    for lab, n in label_counts.items():
        n_use = int(max(0, n))
        if n_use <= 0:
            continue
        idx = np.arange(offset, offset + n_use)
        offset += n_use
        Q = Z_hat[idx]
        Gfor = Z_for[y_for == int(lab)]
        if Gfor.shape[0] == 0:
            continue
        dfor = knn_mean(Q, Gfor, k)
        all_d_for.append(dfor)
        d_knn_forget_mean = float(np.mean(dfor))
        if d_knn_forget_mean > 100:
            d_knn_forget_mean = d_knn_forget_mean / 100
        per_class[int(lab)] = {
            "n_hat": n_use,
            "d_knn_forget_mean": d_knn_forget_mean,
        }

    if not all_d_for:
        return {"per_class": per_class, "overall": None}

    Dfor = np.concatenate(all_d_for)
    d_knn_forget_mean = float(np.mean(Dfor))
    if d_knn_forget_mean > 100:
        d_knn_forget_mean = d_knn_forget_mean / 100
    
    overall = {
        "n_hat_total": int(Dfor.shape[0]),
        "d_knn_forget_mean": d_knn_forget_mean,
    }

    return {"per_class": per_class, "overall": overall}


def evaluate_unlearning(model, test_datasets, exp_dir, config, step=None, removed_dataset=None, val_datasets=None):

    """
    Evaluate model on test datasets.
    
    Args:
        model: The model to evaluate
        test_datasets: Dictionary of test datasets
        exp_dir: Directory to save results
        config: Arguments
        step (int, optional): Current unlearning step (batch index) for filename.
        removed_dataset (tuple, optional): Tuple of (X_remove, y_remove) for MIA evaluation.
        val_datasets (dict, optional): Dictionary of validation datasets for threshold selection.
    
    """
    results = {}
    preds = {}
    step_str = f"_step{step}" if step is not None else "0"
    removed_data, removed_label = removed_dataset

    for ds_name, test_dataset_dict in test_datasets.items():
        val_ds = val_datasets.get(ds_name, None) if val_datasets else None

        # --- cls and fairness evaluation --- 
        results[ds_name], _, preds[ds_name] = evaluate_cls_fairness(
            model, test_dataset_dict, config.device, config.eval_fairness, config.protected_attr,
            val_dataset=val_ds)
        
        step_log_prefix = f"Step {step}" if step is not None else "Initial"
        print(f"{step_log_prefix}, {ds_name} AU-ROC: {results[ds_name]['au-roc']:.4f}")

        if config.eval_fairness:
            fairness_key = f'fairness_{config.protected_attr}'
            print(f"{step_log_prefix}, {ds_name} Fairness for {config.protected_attr}: tpr sd = {results[ds_name][fairness_key]['tpr_sd']:.4f}, fpr sd = {results[ds_name][fairness_key]['fpr_sd']:.4f}, dp sd = {results[ds_name][fairness_key]['demographic_parity_sd']['dp_sd']:.4f}")
        
        # --- MIA evaluation --- 
        if config.eval_mia and removed_data is not None:
            test_data, test_label = test_dataset_dict["test_input"], test_dataset_dict["test_label"]
            mia_res = evaluate_mia(model, removed_data, test_data, removed_label, test_label, config.device, seed=config.mia_kwargs.get('seed', 1234), sample_test_size=config.mia_kwargs.get('sample_test_size', None))
            results[ds_name]['mia_res'] = mia_res
            print(f"{step_log_prefix}, {ds_name} MIA AUC: {mia_res:.4f}")
        
        # --- Model inversion evaluation --- 
        if config.eval_model_inversion and removed_data is not None:
            if config.model_inversion_kwargs.get('inv_per_class', None) is not None:
                unique_labels = np.unique(removed_label.detach().cpu().numpy().reshape(-1)).astype(int)
                label_counts = {int(l): int(config.model_inversion_kwargs['inv_per_class']) for l in unique_labels}
            else:
                label_counts = {0: test_label.shape[0], 1: test_label.shape[0]}

            model_inversion_res = inversion_knn_distance_by_label(model, removed_data, removed_label, config.device, label_counts, seed=config.model_inversion_kwargs.get('seed', 1234), **config.model_inversion_kwargs)
            results[ds_name]['model_inversion_res'] = model_inversion_res
            print(f"{step_log_prefix}, {ds_name} Model Inversion KNN L2 distance: {model_inversion_res['overall']['d_knn_forget_mean']:.4f}")

        # --- Save results --- 
        results_dir = os.path.join(exp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"{ds_name}_results{step_str}.json")
        with open(results_path, 'w') as f:
            json.dump(convert_to_json_serializable(results[ds_name]), f, indent=2)
        
        # --- Save preds --- 
        preds_path = os.path.join(results_dir, f"{ds_name}_preds{step_str}.npy")
        np.save(preds_path, preds[ds_name])

    return results
