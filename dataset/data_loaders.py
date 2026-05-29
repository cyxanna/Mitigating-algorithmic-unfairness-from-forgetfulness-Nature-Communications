import os
import torch
import numpy as np
import pandas as pd
import h5py
import json
import torch.nn.functional as F
from collections import OrderedDict
from sklearn.model_selection import train_test_split


def get_EICU_dataset(data_root, device='cuda:0', task_type='mortality', val_split_seed=42):

    if task_type == "shock" or task_type == "mortality":
        data_path = os.path.join(data_root, f'{task_type}.h5')
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    dec_cat = ['apacheadmissiondx', 'GCS Total', 'Eyes', 'Motor', 'Verbal']
    dec_num = ['admissionheight', 'admissionweight', 'age', 'Heart Rate', 
               'MAP (mmHg)', 'Invasive BP Diastolic', 'Invasive BP Systolic',
               'O2 Saturation', 'Respiratory Rate', 'Temperature (C)', 
               'glucose', 'FiO2', 'pH']
    col_used = ['patientunitstayid'] + dec_cat + dec_num + ['hospitaldischargestatus']

    processed_data = {}
    with h5py.File(data_path, 'r') as f:
        for key in f.keys():
            if len(f[key].shape) == 0:
                processed_data[key] = f[key][()]
            else:
                processed_data[key] = f[key][:]

    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    train_gender = processed_data['train_gender']
    train_ethnicity = processed_data['train_ethnicity']
    train_age = processed_data['train_age_group']
    test_gender = processed_data['test_gender']
    test_ethnicity = processed_data['test_ethnicity']
    test_age = processed_data['test_age_group']

    if y_train.shape[1] == 25:
        y_train = y_train[:, 8]
        y_test = y_test[:, 8]
        print(f"{task_type}")
    else:
        print(f"{task_type}")

    cat_dim = len(dec_cat)
    # One-hot encode categorical features
    X_train_nc = X_train[:, :, cat_dim:]
    X_train_cat = X_train[:, :, :cat_dim].astype(int)
    X_train_cat = F.one_hot(torch.tensor(X_train_cat), num_classes=429)
    X_train_cat = (X_train_cat.sum(dim=-2) > 0).to(torch.int)

    X_test_nc = X_test[:, :, cat_dim:]
    X_test_cat = X_test[:, :, :cat_dim].astype(int)
    X_test_cat = F.one_hot(torch.tensor(X_test_cat), num_classes=429)
    X_test_cat = (X_test_cat.sum(dim=-2) > 0).to(torch.int)

    X_train = np.concatenate([X_train_nc, X_train_cat], axis=2)
    X_train = torch.tensor(X_train).to(device).to(torch.float32)
    X_test = np.concatenate([X_test_nc, X_test_cat], axis=2)
    X_test = torch.tensor(X_test).to(device).to(torch.float32)
    
    y_train = torch.tensor(y_train).to(device).to(torch.float32)
    y_test = torch.tensor(y_test).to(device).to(torch.float32)

    train_gender = torch.tensor(train_gender).to(device).to(torch.int32)
    train_ethnicity = torch.tensor(train_ethnicity).to(device).to(torch.int32)
    train_age = torch.tensor(train_age).to(device).to(torch.int32)
    test_gender = torch.tensor(test_gender).to(device).to(torch.int32)
    test_ethnicity = torch.tensor(test_ethnicity).to(device).to(torch.int32)
    test_age = torch.tensor(test_age).to(device).to(torch.int32)

    # Combine train and test to compute distribution over the full dataset
    combined_gender = torch.cat([train_gender, test_gender], dim=0)
    combined_ethnicity = torch.cat([train_ethnicity, test_ethnicity], dim=0)
    
    # Split validation set from training data (10%, stratified by label)
    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
    val_ratio = 0.125
    n_train = X_train.shape[0]
    idx_all = np.arange(n_train)
    idx_train, idx_val = train_test_split(idx_all, test_size=val_ratio, random_state=val_split_seed, stratify=y_train_np)

    X_val = X_train[idx_val]
    y_val = y_train[idx_val]
    val_gender = train_gender[idx_val]
    val_ethnicity = train_ethnicity[idx_val]
    val_age = train_age[idx_val]

    X_train = X_train[idx_train]
    y_train = y_train[idx_train]
    train_gender = train_gender[idx_train]
    train_ethnicity = train_ethnicity[idx_train]
    train_age = train_age[idx_train]

    # Build training dataset
    dataset={
        "train_input": X_train,
        "train_label": torch.unsqueeze(y_train,1),
        "train_protected_gender": train_gender,
        "train_protected_ethnicity": train_ethnicity,
        "train_samples": X_train.shape[0],
        "train_samples_positive": torch.sum(y_train == 1.0).item(),
    }

    # Build validation dataset
    val_dataset = {
        "test_input": X_val,
        "test_label": torch.unsqueeze(y_val, 1),
        "test_protected_gender": val_gender,
        "test_protected_ethnicity": val_ethnicity,
        "test_samples": X_val.shape[0],
        "test_samples_positive": torch.sum(y_val == 1.0).item(),
    }

    # Build test dataset  
    test_dataset = {
        "test_input": X_test,
        "test_label": torch.unsqueeze(y_test,1),
        "test_protected_gender": test_gender,
        "test_protected_ethnicity": test_ethnicity,
        "test_samples": X_test.shape[0],
        "test_samples_positive": torch.sum(y_test == 1.0).item(),
    }
    
    meta_info = {
        "dataset_name": "EICU",
        "train_samples": X_train.shape[0],
        "val_samples": X_val.shape[0],
        "test_samples": X_test.shape[0],
        "fea_dim": X_train.shape[2],
        "train_samples_positive": torch.sum(y_train == 1.0).item(),
        "val_samples_positive": torch.sum(y_val == 1.0).item(),
        "test_samples_positive": torch.sum(y_test == 1.0).item(),
        "input_features": col_used,
        "task_type": task_type,
        "sequence_length": X_train.shape[1],
    }

    return [dataset, val_dataset, test_dataset, meta_info]



# MIMIC ethnicity mapping
UNKNOWN_SET = {'UNKNOWN', 'UNABLE TO OBTAIN', 'PATIENT DECLINED TO ANSWER'}
BLACK_KEYS = {'BLACK/AFRICAN AMERICAN','BLACK/AFRICAN','BLACK/CAPE VERDEAN','BLACK/CARIBBEAN ISLAND'}
ASIAN_KEYS = {'ASIAN - ASIAN INDIAN','ASIAN - SOUTH EAST ASIAN', 'ASIAN - CHINESE', 'ASIAN', 'ASIAN - KOREAN'}
WHITE_KEYS = {'WHITE', 'WHITE - RUSSIAN','WHITE - OTHER EUROPEAN','WHITE - EASTERN EUROPEAN', 'WHITE - BRAZILIAN'}
HISPANIC_KEYS = {'HISPANIC/LATINO - PUERTO RICAN', 'HISPANIC OR LATINO', 'HISPANIC/LATINO - DOMINICAN', 'HISPANIC/LATINO - GUATEMALAN', 'HISPANIC/LATINO - SALVADORAN', 'HISPANIC/LATINO - CUBAN', 'HISPANIC/LATINO - MEXICAN',  'HISPANIC/LATINO - COLUMBIAN', 'HISPANIC/LATINO - CENTRAL AMERICAN', 'HISPANIC/LATINO - HONDURAN'}
ETHNICITY_MAP_MIMIC = {'White': 0, 'Asian': 1, 'Black': 2, 'Hispanic': 3, 'Other': 4, 'Unknown': 5}
R_ETHNICITY_MAP_MIMIC = {v: k for k, v in ETHNICITY_MAP_MIMIC.items()}

def fine_label_to_coarse(name: str) -> str:
    if name is None or str(name).strip() == '' or str(name).upper() in UNKNOWN_SET:
        return 'Unknown'
    s = str(name).upper()
    if s in BLACK_KEYS:
        return 'Black'
    if s in ASIAN_KEYS:
        return 'Asian'
    if s in WHITE_KEYS:
        return 'White'
    if s in HISPANIC_KEYS:
        return 'Hispanic'
    return 'Other'

def get_MIMIC_dataset(data_root, device='cuda:0', val_split_seed=42, test_split_seed=1019):

    # 1) Load outputs from build_npz.py
    data_path = os.path.join(data_root, 'dataset_X.npz')
    label_path = os.path.join(data_root, 'dataset_Y.npz')
    meta_path = os.path.join(data_root, 'dataset_metadata.json')

    data = np.load(data_path)
    label = np.load(label_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    charts = data['chart']  # (N, T, Vc) or (N, 0, 0)
    outs   = data['out']    # (N, T, Vo) or (N, 0, 0)
    stats  = data['stat']   # (N, Vs)
    demos  = data['demo']   # (N, 4)
    y      = label['y']      # (N,)

    N = int(y.shape[0])
    # Infer global T (already padded during build)
    T = int(charts.shape[1] if charts.shape[1] > 0 else outs.shape[1])

    # 2) Assemble (N, T, D) features by concatenating along the feature dimension
    feats = []
    if charts.shape[1] > 0 and charts.shape[2] > 0:
        feats.append(charts.astype(np.float32, copy=False))
    if outs.shape[1] > 0 and outs.shape[2] > 0:
        feats.append(outs.astype(np.float32, copy=False))
    if stats.shape[1] > 0:
        # Repeat static features along the time dimension
        stat_rep = np.repeat(stats[:, None, :].astype(np.float32, copy=False), T, axis=1)  # (N, T, Vs)
        feats.append(stat_rep)

    X_all = np.concatenate(feats, axis=2)  # (N, T, D_total)

    # Protected attributes (for distribution stats and downstream evaluation)
    PG = demos[:, 0].astype(np.int32, copy=False)  # gender
    PE = demos[:, 1].astype(np.int32, copy=False)  # ethnicity (fine-grained)

    # First map ethnicity from fine to coarse codes and update PE for later use
    ev = meta['demo_vocabs'].get('ethnicity_vocab', {})  # name -> fine code
    r_ev = {int(v): k for k, v in ev.items()}            # fine code -> name
    coarse_names = [fine_label_to_coarse(r_ev.get(int(code), None)) for code in PE]
    PE = np.asarray([ETHNICITY_MAP_MIMIC[name] for name in coarse_names], dtype=np.int32)

    # 3) Split train+val / test, then train / val
    (X_tv, X_te, y_tv, y_te, PG_tv, PG_te, PE_tv, PE_te) = train_test_split(
        X_all, y, PG, PE, test_size=0.2, random_state=test_split_seed, stratify=y)
    (X_tr, X_va, y_tr, y_va, PG_tr, PG_va, PE_tr, PE_va) = train_test_split(
        X_tv, y_tv, PG_tv, PE_tv, test_size=0.09, random_state=val_split_seed, stratify=y_tv)

    # 4) Convert to tensor
    def to_dev(arr, dtype): return torch.tensor(arr, dtype=dtype, device=device)

    train_input = to_dev(X_tr, torch.float32)
    train_label = to_dev(y_tr, torch.float32)
    val_input   = to_dev(X_va, torch.float32)
    val_label   = to_dev(y_va, torch.float32)
    test_input  = to_dev(X_te, torch.float32)
    test_label  = to_dev(y_te, torch.float32)

    train_PG = to_dev(PG_tr, torch.int32)
    train_PE = to_dev(PE_tr, torch.int32)
    val_PG   = to_dev(PG_va, torch.int32)
    val_PE   = to_dev(PE_va, torch.int32)
    test_PG  = to_dev(PG_te, torch.int32)
    test_PE  = to_dev(PE_te, torch.int32)

    train_dataset = {
        "train_input": train_input,
        "train_label": train_label,
        "train_protected_gender": train_PG,
        "train_protected_ethnicity": train_PE,
        "train_samples": train_input.shape[0],
    }

    val_dataset = {
        "test_input": val_input,
        "test_label": torch.unsqueeze(val_label, 1),
        "test_protected_gender": val_PG,
        "test_protected_ethnicity": val_PE,
        "test_samples": val_input.shape[0],
        "test_samples_positive": torch.sum(val_label == 1.0).item(),
    }

    test_dataset = {
        "test_input": test_input,
        "test_label": torch.unsqueeze(test_label, 1),
        "test_protected_gender": test_PG,
        "test_protected_ethnicity": test_PE,
        "test_samples": test_input.shape[0],
        "test_samples_positive": torch.sum(test_label == 1.0).item(),
    }

    meta_info = {
        "dataset_name": "MIMIC",
        "train_samples": train_input.shape[0],
        "val_samples":   val_input.shape[0],
        "test_samples":  test_input.shape[0],
        "total_samples": N,
        "seq_len":       T,
        "fea_dim":       train_input.shape[2],
        "train_samples_positive": torch.sum(train_label == 1.0).item(),
        "val_samples_positive":   torch.sum(val_label == 1.0).item(),
        "test_samples_positive":  torch.sum(test_label == 1.0).item(),
    }

    return [train_dataset, val_dataset, test_dataset, meta_info]


def get_your_private_dataset(data_root):
    # Your loading and processing code here

    train_dataset = {
        "train_input": None,
        "train_label": None,
        "train_protected_gender": None,
        "train_protected_ethnicity": None,
        "train_samples": None,
    }

    val_dataset = {
        "test_input": None,
        "test_label": None,
        "test_protected_gender": None,
        "test_protected_ethnicity": None,
        "test_samples": None,
    }

    test_dataset = {
        "test_input": None,
        "test_label": None,
        "test_protected_gender": None,
        "test_protected_ethnicity": None,
        "test_samples": None,
    }

    meta_info = {
        "dataset_name": "YOUR PRIVATE DATASET",
        "train_samples": None,
        "val_samples":   None,
        "test_samples":  None,
        "total_samples": None,
        "seq_len":       None,
        "fea_dim":       None,
        "train_samples_positive": None,
        "val_samples_positive": None,
        "test_samples_positive": None,
    }

    return [train_dataset, val_dataset, test_dataset, meta_info]

DATASET_FUNCS = {
    "EICU": get_EICU_dataset,
    "MIMIC": get_MIMIC_dataset,
    "YOUR PRIVATE DATASET": get_your_private_dataset,
}

def get_dataset(
        dataset_name,  # dataset_name
        **kwargs
):
    ## warpper function for loading datasets
    ds = dataset_name.strip().upper()
    assert ds in DATASET_FUNCS, f"Data loader not defined for dataset: {ds}  "
    return DATASET_FUNCS[ds](**kwargs)
