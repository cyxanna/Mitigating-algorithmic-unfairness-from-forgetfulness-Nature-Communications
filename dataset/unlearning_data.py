import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import logging

def stratified_sample(indices, labels, n_samples):
    if len(indices) == 0:
        return np.array([], dtype=int)
    if n_samples <= 0:
            return np.array([], dtype=int)
    n_samples = min(n_samples, len(indices))

    pos_indices = indices[labels[indices] == 1]
    neg_indices = indices[labels[indices] == 0]

    if len(pos_indices) == 0 and len(neg_indices) == 0:
        return np.array([], dtype=int)
    elif len(pos_indices) == 0:
        pos_ratio = 0.0
    elif len(neg_indices) == 0:
            pos_ratio = 1.0
    else:
        pos_ratio = len(pos_indices) / len(indices)

    n_pos = int(n_samples * pos_ratio)
    n_neg = n_samples - n_pos

    n_pos = min(n_pos, len(pos_indices))
    n_neg = min(n_neg, len(neg_indices))

    if n_pos + n_neg < n_samples:
        if len(pos_indices) > n_pos:
                n_pos = min(len(pos_indices), n_samples - n_neg)
        elif len(neg_indices) > n_neg:
                n_neg = min(len(neg_indices), n_samples - n_pos)

    sampled_pos = np.random.choice(pos_indices, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
    sampled_neg = np.random.choice(neg_indices, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)

    sampled_indices = np.concatenate([sampled_pos, sampled_neg])
    np.random.shuffle(sampled_indices)

    return sampled_indices


def prepare_unlearning_data_by_ratio(dataset, config):
    """Prepare data for unlearning based on removal ratio, incrementing step by step.

    Args:
        dataset: Dictionary containing training data and attributes.
        config: Arguments containing unlearning configs like:
            - remove_ratio (float): The target total ratio of data to remove (e.g., 0.1 for 10%).
            - step_ratio (float): The ratio increment per step (e.g., 0.01 for 1%).
            - sample_method (str): 'random', 'minority', 'majority'.
            - protected_attr (str): 'gender', 'ethnicity', 'age', 'ethnicity_age', 'hospital_site'.
            - batch_size (int): DataLoader batch size for remaining data.
            - accumulate_removed (bool): If True, removed_loader contains all data removed so far.
                                         If False, removed_loader contains only the data for the current step.
                                         Remaining loader logic depends on this too.
            - seed (int, optional): Random seed.
            - dataset_name (str): Name of the dataset.

    Returns:
        indices_to_remove: Pre-selected indices for the entire removal process.
        initial_remaining_loader: DataLoader for the initial full dataset.
        get_batch_loaders: Function to get removed and remaining loaders for each step.
        num_steps: The total number of removal steps.
        step_remove_size: The number of samples removed in each step.
    """

    if config.dataset_name.upper() == "BH":
        raise ValueError("BH dataset cannot be used for unlearning (evaluation only)")

    protected_attr = config.protected_attr

    X_train = dataset["train_input"]
    y_train = dataset["train_label"].squeeze()
    key_name = f'train_protected_{protected_attr}'

    if key_name in dataset:
        protected_attr_data = dataset[key_name]
    else:
        raise ValueError(f"Protected attribute {protected_attr} not found in dataset")

    if X_train is None or y_train is None:
        raise ValueError(f"Invalid dataset format for {config.dataset_name}")

    total_samples = len(X_train)

    total_num_removes = int(total_samples * config.remove_ratio)
    if total_num_removes <= 0:
        logging.warning(f"Calculated total_num_removes is {total_num_removes}. No samples will be removed.")

    step_remove_size = int(total_samples * config.step_ratio)
    if step_remove_size <= 0:
        raise ValueError("step_ratio results in zero samples per step. Increase step_ratio.")

    num_steps = (total_num_removes + step_remove_size - 1) // step_remove_size
    actual_total_removed = num_steps * step_remove_size

    logging.info(f"Total samples: {total_samples}")
    logging.info(f"Target remove ratio: {config.remove_ratio:.2%}, Target remove count: {total_num_removes}")
    logging.info(f"Step ratio: {config.step_ratio:.2%}, Step remove size: {step_remove_size}")
    logging.info(f"Total steps: {num_steps}")
    logging.info(f"Actual total samples to be selected for removal: {actual_total_removed}")

    if config.sample_method == 'random':
        indices_to_remove = stratified_sample(
            np.arange(total_samples),
            y_train.cpu().numpy(),
            actual_total_removed
        )
    else:
        unique_values, counts = torch.unique(protected_attr_data, return_counts=True)

        if config.sample_method == 'majority':
            target_value = unique_values[counts.argmax()]
            mask = (protected_attr_data == target_value).cpu().numpy()
        else:
            majority_value = unique_values[counts.argmax()]
            mask = (protected_attr_data != majority_value).cpu().numpy()

        eligible_indices = np.where(mask)[0]
        num_available_eligible = len(eligible_indices)

        indices_to_remove = stratified_sample(
            eligible_indices,
            y_train.cpu().numpy(),
            min(actual_total_removed, num_available_eligible)
        )

        if len(indices_to_remove) < actual_total_removed:
            logging.warning(f"Warning: Only {len(indices_to_remove)} samples available in target group(s) for removal, "
                            f"less than calculated total {actual_total_removed}. Adjusting num_steps.")
            actual_total_removed = len(indices_to_remove)
            num_steps = (actual_total_removed + step_remove_size - 1) // step_remove_size


    np.random.shuffle(indices_to_remove)

    initial_remaining_loader = DataLoader(
        TensorDataset(
            X_train,
            y_train,
            protected_attr_data,
        ),
        batch_size=config.batch_size,
        shuffle=True
    )

    def get_batch_loaders(step_idx):
        """Get removed and remaining loaders for current step.

        Args:
            step_idx: Current step index (0 to num_steps-1).

        Returns:
            removed_loader: DataLoader for samples corresponding to this step or accumulated.
            remaining_loader: DataLoader for remaining samples (data not removed *up to this step*).
        """
        if step_idx >= num_steps:
             raise IndexError(f"step_idx {step_idx} is out of bounds for num_steps {num_steps}")

        cumulative_end_idx = min((step_idx + 1) * step_remove_size, len(indices_to_remove))
        cumulative_removed_indices = indices_to_remove[:cumulative_end_idx]

        step_start_idx = step_idx * step_remove_size
        current_removed_indices_for_loader = indices_to_remove[step_start_idx:cumulative_end_idx]
        removed_batch_size = step_remove_size

        all_remaining_indices = np.setdiff1d(
            np.arange(total_samples),
            cumulative_removed_indices
        )

        if len(current_removed_indices_for_loader) > 0:
            removed_loader = DataLoader(
                TensorDataset(
                    X_train[current_removed_indices_for_loader],
                    y_train[current_removed_indices_for_loader],
                    protected_attr_data[current_removed_indices_for_loader]
                ),
                batch_size=min(removed_batch_size, len(current_removed_indices_for_loader)),
                shuffle=True
            )
        else:
            removed_loader = None # No data to remove in this step/accumulation

        if len(all_remaining_indices) > 0:
            remaining_loader = DataLoader(
                TensorDataset(
                    X_train[all_remaining_indices],
                    y_train[all_remaining_indices],
                    protected_attr_data[all_remaining_indices]
                ),
                batch_size=config.batch_size,
                shuffle=True
            )
        else:
            remaining_loader = None
            logging.warning(f"Step {step_idx+1}: Remaining loader has no data.")


        return removed_loader, remaining_loader

    return indices_to_remove, initial_remaining_loader, get_batch_loaders, num_steps, step_remove_size 


def compute_step_plan(total_samples, config):
    """Compute per-step removal size and total steps based on config.

    Args:
        total_samples (int): Total number of training samples.
        config: Config object with remove_ratio and step_ratio.

    Returns:
        (step_remove_size, num_steps)
    """
    total_num_removes = int(total_samples * config.remove_ratio)
    step_remove_size = int(total_samples * config.step_ratio)
    if step_remove_size <= 0:
        raise ValueError("step_ratio results in zero samples per step. Increase step_ratio.")
    num_steps = (total_num_removes + step_remove_size - 1) // step_remove_size
    return step_remove_size, num_steps


def sample_removed_from_remaining(
    remaining_indices,
    y_train,
    protected_attr_data,
    step_remove_size,
    sample_method,
    seed,
    target_protected_values=None,
    label_sampling: str = "stratified",
):
    if isinstance(y_train, torch.Tensor):
        y_np = y_train.detach().cpu().numpy().flatten()
    else:
        y_np = np.asarray(y_train).flatten()
    if isinstance(protected_attr_data, torch.Tensor):
        prot_np = protected_attr_data.detach().cpu().numpy().flatten()
    else:
        prot_np = np.asarray(protected_attr_data).flatten()

    remaining_indices = np.asarray(remaining_indices, dtype=int)
    if len(remaining_indices) == 0:
        return np.array([], dtype=int), remaining_indices.copy()

    rng = np.random.default_rng(seed)

    def _stratified_on_subset(candidates):
        if len(candidates) == 0:
            return np.array([], dtype=int)
        labels = y_np[candidates]
        pos_mask = labels == 1
        pos_indices = candidates[pos_mask]
        neg_indices = candidates[~pos_mask]
        if len(pos_indices) == 0 and len(neg_indices) == 0:
            return np.array([], dtype=int)
        if len(pos_indices) == 0:
            pos_ratio = 0.0
        elif len(neg_indices) == 0:
            pos_ratio = 1.0
        else:
            pos_ratio = len(pos_indices) / len(candidates)
        n_pos = int(n_samples * pos_ratio)
        n_neg = n_samples - n_pos
        n_pos = min(n_pos, len(pos_indices))
        n_neg = min(n_neg, len(neg_indices))
        if n_pos + n_neg < n_samples:
            if len(pos_indices) > n_pos:
                n_pos = min(len(pos_indices), n_samples - n_neg)
            elif len(neg_indices) > n_neg:
                n_neg = min(len(neg_indices), n_samples - n_pos)
        sampled_pos = rng.choice(pos_indices, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
        sampled_neg = rng.choice(neg_indices, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)
        sampled = np.concatenate([sampled_pos, sampled_neg])
        rng.shuffle(sampled)
        return sampled

    rem_prot = prot_np[remaining_indices]

    if target_protected_values is not None:
        allowed = np.array(list(target_protected_values))
        eligible_mask = np.isin(rem_prot, allowed)
        eligible_indices = remaining_indices[eligible_mask]
    elif sample_method == 'random':
        eligible_indices = remaining_indices
    else:
        unique_values, counts = np.unique(rem_prot, return_counts=True)
        majority_value = unique_values[counts.argmax()]
        eligible_mask = (rem_prot == majority_value) if sample_method == 'majority' else (rem_prot != majority_value)
        eligible_indices = remaining_indices[eligible_mask]

    if label_sampling == "neg_only":
        eligible_indices = eligible_indices[y_np[eligible_indices] == 0]
    elif label_sampling == "pos_only":
        eligible_indices = eligible_indices[y_np[eligible_indices] == 1]

    n_samples = int(min(step_remove_size, len(eligible_indices)))
    if n_samples <= 0 or len(eligible_indices) == 0:
        return np.array([], dtype=int), remaining_indices.copy()

    if label_sampling == "stratified":
        removed_indices = _stratified_on_subset(eligible_indices)
    else:
        removed_indices = rng.choice(eligible_indices, size=n_samples, replace=False)

    removed_set = set(removed_indices.tolist())
    new_remaining_indices = np.array([idx for idx in remaining_indices if idx not in removed_set], dtype=int)
    return removed_indices.astype(int), new_remaining_indices.astype(int)

def build_loaders_by_indices(X, y, prot, removed_indices, remaining_indices, batch_size):
    """Build DataLoaders given global indices for removed and remaining pools.

    Args:
        X (torch.Tensor): All training inputs.
        y (torch.Tensor): All training labels.
        prot (torch.Tensor): All training protected attribute values.
        removed_indices (np.ndarray): Global indices of removed set for this step.
        remaining_indices (np.ndarray): Global indices of remaining set after removal.
        batch_size (int): Batch size for remaining loader.

    Returns:
        (removed_loader, remaining_loader)
    """
    removed_loader = None
    remaining_loader = None

    if removed_indices is not None and len(removed_indices) > 0:
        removed_loader = DataLoader(
            TensorDataset(
                X[removed_indices],
                y[removed_indices],
                prot[removed_indices]
            ),
            batch_size=max(1, int(len(removed_indices))),
            shuffle=True
        )

    if remaining_indices is not None and len(remaining_indices) > 0:
        remaining_loader = DataLoader(
            TensorDataset(
                X[remaining_indices],
                y[remaining_indices],
                prot[remaining_indices]
            ),
            batch_size=batch_size,
            shuffle=True
        )

    return removed_loader, remaining_loader