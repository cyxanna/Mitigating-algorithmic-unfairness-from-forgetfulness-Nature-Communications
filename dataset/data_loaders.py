import os
import torch
import numpy as np
import pandas as pd
import h5py
import json
import torch.nn.functional as F
from collections import OrderedDict
from sklearn.model_selection import train_test_split


def get_EICU_dataset(data_root, device='cuda:0', task_type='mortality', intersectional=False, cal_dist=False):

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
        # Load all datasets
        for key in f.keys():
            if len(f[key].shape) == 0:  # Scalar data
                processed_data[key] = f[key][()]
            else:  # Array data
                processed_data[key] = f[key][:]

    # cat_unique_value = processed_data['n_cat_class']
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

    # Create ethnicity_age cross attribute when needed
    train_ethnicity_age = None
    test_ethnicity_age = None
    
    if intersectional:
        # Build an EICU-specific ethnicity_age mapping
        eicu_ethnicity_age_map = {}
        map_idx = 0
        for eth in range(6):  # EICU ethnicity: 0-5 
            for age in range(3):  # Age groups: 0-2
                eicu_ethnicity_age_map[f"{eth}_{age}"] = map_idx
                map_idx += 1
        
        def create_ethnicity_age_code(eth_codes, age_codes):
            """Create EICU ethnicity_age cross codes"""
            eth_age_codes = []
            for eth, age in zip(eth_codes, age_codes):
                # Use the original EICU ethnicity code without remapping
                eth_age_key = f"{eth}_{age}"
                if eth_age_key in eicu_ethnicity_age_map:
                    eth_age_codes.append(eicu_ethnicity_age_map[eth_age_key])
                else:
                    eth_age_codes.append(-1)  # Unknown combination
            
            return np.array(eth_age_codes)
        
        train_ethnicity_age_codes = create_ethnicity_age_code(train_ethnicity.cpu().numpy(), train_age.cpu().numpy())
        test_ethnicity_age_codes = create_ethnicity_age_code(test_ethnicity.cpu().numpy(), test_age.cpu().numpy())
        
        train_ethnicity_age = torch.tensor(train_ethnicity_age_codes).to(device).to(torch.int32)
        test_ethnicity_age = torch.tensor(test_ethnicity_age_codes).to(device).to(torch.int32)

    # Combine train and test to compute distribution over the full dataset
    combined_gender = torch.cat([train_gender, test_gender], dim=0)
    combined_ethnicity = torch.cat([train_ethnicity, test_ethnicity], dim=0)
    
    # Gender distribution
    if cal_dist:
        print(f"\n{task_type.upper()} Gender Distribution:")
        gender_unique, gender_counts = torch.unique(combined_gender, return_counts=True)
        gender_map = {0: 'Female', 1: 'Male'}
        total_samples = combined_gender.shape[0]
        
        for code, count in zip(gender_unique.cpu().numpy(), gender_counts.cpu().numpy()):
            gender_name = gender_map.get(code, f'Unknown_{code}')
            ratio = (count / total_samples * 100)
            print(f"{gender_name} ({code}): {count} ({ratio:.1f}%)")
        
        # Ethnicity distribution
        print(f"\n{task_type.upper()} Ethnicity Distribution:")
        ethnicity_unique, ethnicity_counts = torch.unique(combined_ethnicity, return_counts=True)
        ethnicity_map = {0: 'NaN', 1: 'Asian', 2: 'African American', 3: 'Caucasian', 4: 'Hispanic', 5: 'Native American'}
        
        for code, count in zip(ethnicity_unique.cpu().numpy(), ethnicity_counts.cpu().numpy()):
            ethnicity_name = ethnicity_map.get(code, f'Unknown_{code}')
            ratio = (count / total_samples * 100)
            print(f"{ethnicity_name} ({code}): {count} ({ratio:.1f}%)")
        
    # Age distribution when in intersectional mode
    if intersectional:
        combined_age = torch.cat([train_age, test_age], dim=0)
        print(f"\n{task_type.upper()} Age_Group Distribution:")
        age_unique, age_counts = torch.unique(combined_age, return_counts=True)
        age_map = {1: '18-39', 2: '40-64', 3: '65+'}
        
        for code, count in zip(age_unique.cpu().numpy(), age_counts.cpu().numpy()):
            age_name = age_map.get(code, f'Unknown_{code}')
            ratio = (count / total_samples * 100)
            print(f"{age_name} ({code}): {count} ({ratio:.1f}%)")
        
        # Ethnicity_Age joint distribution
        if train_ethnicity_age is not None and test_ethnicity_age is not None:
            combined_ethnicity_age = torch.cat([train_ethnicity_age, test_ethnicity_age], dim=0)
            print(f"\n{task_type.upper()} Ethnicity_Age Distribution:")
            eth_age_unique, eth_age_counts = torch.unique(combined_ethnicity_age, return_counts=True)
            
            # Build reverse mapping for EICU codes
            reverse_eicu_eth_age_map = {v: k for k, v in eicu_ethnicity_age_map.items()}
            eicu_ethnicity_map = {0: 'NaN', 1: 'Asian', 2: 'African American', 3: 'Caucasian', 4: 'Hispanic', 5: 'Native American'}
            eicu_age_map = {0: '18-39', 1: '40-64', 2: '65+'}
            
            for code, count in zip(eth_age_unique.cpu().numpy(), eth_age_counts.cpu().numpy()):
                if code == -1:
                    combo_name = "Unknown_Combo"
                elif code in reverse_eicu_eth_age_map:
                    eth_str, age_str = reverse_eicu_eth_age_map[code].split('_')
                    eth_name = eicu_ethnicity_map.get(int(eth_str), f'Eth_{eth_str}')
                    age_name = eicu_age_map.get(int(age_str), f'Age_{age_str}')
                    combo_name = f"{eth_name}_{age_name}"
                else:
                    combo_name = f"Code_{code}"
                
                ratio = (count / total_samples * 100)
                print(f"{combo_name} ({code}): {count} ({ratio:.1f}%)")

    # Build training dataset
    dataset={
        "train_input": X_train,
        "train_label": torch.unsqueeze(y_train,1),
        "train_protected_gender": train_gender,
        "train_protected_ethnicity": train_ethnicity,
        "train_samples": X_train.shape[0],
        "train_samples_positive": torch.sum(y_train == 1.0).item(),
    }
    
    if intersectional:
        # Add age-related sensitive attributes
        dataset.update({
            "train_protected_age": train_age,
            "train_protected_ethnicity_age": train_ethnicity_age,
        })

    # Build test dataset  
    test_dataset = {
        "test_input": X_test,
        "test_label": torch.unsqueeze(y_test,1),
        "test_protected_gender": test_gender,
        "test_protected_ethnicity": test_ethnicity,
        "test_samples": X_test.shape[0],
        "test_samples_positive": torch.sum(y_test == 1.0).item(),
    }
    
    if intersectional:
        # Add age-related sensitive attributes
        test_dataset.update({
            "test_protected_age": test_age,
            "test_protected_ethnicity_age": test_ethnicity_age,
        })

    meta_info = {
        "dataset_name": "EICU-V2_INTER" if intersectional else "EICU-V2",
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "fea_dim": X_train.shape[2],  # Feature dimension (excluding time and batch axes)
        "train_samples_positive": torch.sum(y_train == 1.0).item(),
        "test_samples_positive": torch.sum(y_test == 1.0).item(),
        "input_features": col_used,
        "task_type": task_type,
        "sequence_length": X_train.shape[1],  # Temporal length
    }

    return [dataset, test_dataset, meta_info]

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

def get_MIMIC_dataset(data_root, device='cuda:0', cal_dist=True, intersectional=False):

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

    if cal_dist:
        # Reverse vocab mappings for gender only (ethnicity already coarse-coded)
        gv = meta['demo_vocabs'].get('gender_vocab', {})
        r_gv = {v: k for k, v in gv.items()}

        # counts
        def count_and_map(arr, rev_map):
            vals, counts = np.unique(arr, return_counts=True)
            items = [(rev_map.get(int(v), str(int(v))), int(c)) for v, c in zip(vals, counts)]
            # sort by count desc
            items.sort(key=lambda x: x[1], reverse=True)
            return OrderedDict(items)

        gender_dist = count_and_map(PG, r_gv)
        ethnicity_dist = count_and_map(PE, R_ETHNICITY_MAP_MIMIC)

        print('Gender distribution:')
        for k, v in gender_dist.items():
            print(f'  {k}: {v}')
        print('Ethnicity distribution (coarse):')
        N = sum(ethnicity_dist.values())
        for k, v in ethnicity_dist.items():
            print(f'  {k}: {v}, ratio: {v / N * 100:.1f}%')
        print(f'Total samples: {N}')

    # 3) Split train/test while keeping temporal structure unchanged
    (X_tr, X_te, y_tr, y_te, PG_tr, PG_te, PE_tr, PE_te) = train_test_split(
        X_all, y, PG, PE, test_size=0.2, random_state=1019, stratify=y)
    # 4) Convert to tensor
    def to_dev(arr, dtype): return torch.tensor(arr, dtype=dtype, device=device)

    train_input = to_dev(X_tr, torch.float32)      # (B, T, D)
    train_label = to_dev(y_tr, torch.float32)      # (B,)
    test_input  = to_dev(X_te, torch.float32)
    test_label  = to_dev(y_te, torch.float32)

    train_PG = to_dev(PG_tr, torch.int32)
    train_PE = to_dev(PE_tr, torch.int32)
    test_PG  = to_dev(PG_te, torch.int32)
    test_PE  = to_dev(PE_te, torch.int32)

    train_samples = train_input.shape[0]
    test_samples = test_input.shape[0]
    fea_dim = train_input.shape[2]
    train_samples_positive = torch.sum(train_label == 1.0).item()
    test_samples_positive = torch.sum(test_label == 1.0).item()

    train_dataset = {
        "train_input": train_input,
        "train_label": train_label,
        "train_protected_gender": train_PG,
        "train_protected_ethnicity": train_PE,
        "train_samples": train_input.shape[0],
    }

    test_dataset = {
        "test_input": test_input,
        "test_label": torch.unsqueeze(test_label, 1),
        "test_protected_gender": test_PG,
        "test_protected_ethnicity": test_PE,
    }

    meta_info = {
        "dataset_name": "MIMIC",
        "train_samples": train_samples,
        "test_samples":  test_samples,
        "total_samples": N,
        "seq_len":       T,
        "fea_dim":       fea_dim,
        "train_samples_positive": train_samples_positive,
        "test_samples_positive": test_samples_positive,
    }

    return [train_dataset, test_dataset, meta_info]


def get_your_private_dataset(data_root):
    # Your loading and processing code here

    train_dataset = {
        "train_input": None,
        "train_label": None,
        "train_protected_gender": None,
        "train_protected_ethnicity": None,
        "train_samples": None,
    }

    test_dataset = {
        "test_input": None,
        "test_label": None,
        "test_protected_gender": None,
        "test_protected_ethnicity": None,
    }

    meta_info = {
        "dataset_name": "YOUR PRIVATE DATASET",
        "train_samples": None,
        "test_samples":  None,
        "total_samples": None,
        "seq_len":       None,
        "fea_dim":       None,
        "train_samples_positive": None,
        "test_samples_positive": None,
    }

    return [train_dataset, test_dataset, meta_info]

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
