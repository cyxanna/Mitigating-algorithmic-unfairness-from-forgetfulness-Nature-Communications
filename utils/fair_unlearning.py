import torch

from .utils import gram_schmidt_process,flatten_grads, cosine_similarity
import logging
import copy
import random
from torch.utils.data import TensorDataset, DataLoader
import itertools
import numpy as np


def compute_equalized_odds_loss(logits, labels, protected_attr):
    """
    Compute fairness loss that directly optimizes Equalized Odds (TPR and FPR equality)
    by minimizing prediction differences within same-label groups across different sensitive groups
    """
    device = logits.device
    unique_groups = torch.unique(protected_attr)
    unique_labels = torch.unique(labels)
    
    if len(unique_groups) < 2:
        return torch.tensor(0.0, device=device, requires_grad=True), {}
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    stats = {'pos_label_loss': 0.0, 'neg_label_loss': 0.0, 'group_stats': {}}
    
    # For each label value (0 and 1)
    for label in unique_labels:
        label_mask = (labels == label)
        
        # Collect group predictions for this label
        group_preds = {}
        group_sizes = {}
        
        for group in unique_groups:
            group_label_mask = label_mask & (protected_attr == group)
            if group_label_mask.sum() > 0:
                group_preds[group.item()] = logits[group_label_mask]
                group_sizes[group.item()] = group_label_mask.sum().item()
        
        # Skip if less than 2 groups have samples for this label
        if len(group_preds) < 2:
            continue
            
        pairwise_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_pairs = 0
        
        group_ids = list(group_preds.keys())
        for i in range(len(group_ids)):
            for j in range(i + 1, len(group_ids)):
                group_a_preds = group_preds[group_ids[i]]
                group_b_preds = group_preds[group_ids[j]]
                
                # Compute mean difference between groups for this label
                mean_diff = torch.abs(group_a_preds.mean() - group_b_preds.mean())
                pairwise_loss = pairwise_loss + mean_diff
                num_pairs += 1
        
        if num_pairs > 0:
            avg_pairwise_loss = pairwise_loss / num_pairs
            total_loss = total_loss + avg_pairwise_loss
            
            if label.item() == 1:
                stats['pos_label_loss'] = avg_pairwise_loss.item()
            else:
                stats['neg_label_loss'] = avg_pairwise_loss.item()
    
        # Store group statistics for this label
        stats['group_stats'][f'label_{label.item()}'] = {
            group_id: {'mean_pred': preds.mean().item(), 'size': group_sizes[group_id]} 
            for group_id, preds in group_preds.items()
        }
    
    return total_loss, stats


def fair_unlearning(model, unlearning_loader, remaining_loader, loss_fn, config, step):
    """
    Fair unlearning - computes fresh forget gradients each batch
    
    Args:
        model: The model to update
        unlearning_loader: DataLoader for unlearning data
        remaining_loader: DataLoader for remaining data
        loss_fn: Loss function
        config: Arguments containing hyperparameters
        step: Current unlearning step
    """
    device = next(model.parameters()).device
    X_remove, y_remove, protected_attr_remove = next(iter(unlearning_loader))
    X_remove = X_remove.to(device)
    y_remove = y_remove.to(device)
    protected_attr_remove = protected_attr_remove.to(device)

    # Training loop over remaining data
    model.train()
    total_stats = {
        'loss_forget': 0.0,
        'loss_remain': 0.0, 
        'fair_loss': 0.0,
        'batch_count': 0
    }
    
    for iteration in range(config.unlearning_iterations):
        batch_count = 0
        
        for batch_idx, (X_remain, y_remain, protected_attr) in enumerate(remaining_loader):
            X_remain = X_remain.to(device)
            y_remain = y_remain.to(device)
            protected_attr = protected_attr.to(device)
            
            # 1. Compute fresh forget gradient for current batch
            model.eval()
            with torch.enable_grad():
                outputs_remove = model(X_remove)
                if isinstance(outputs_remove, dict):
                    preds_remove = outputs_remove['preds']
                else:
                    preds_remove = outputs_remove
                
                loss_forget = loss_fn(preds_remove.squeeze(), y_remove)
                grad_forget = torch.autograd.grad(loss_forget, model.parameters(), create_graph=False)
                grad_forget = [-g for g in grad_forget]  # Negate for maximization (unlearning)
            
            model.train()
            
            # 2. Compute remain gradient for current batch
            with torch.enable_grad():
                outputs_remain = model(X_remain)
                if isinstance(outputs_remain, dict):
                    logits_remain = outputs_remain['logits']
                    preds_remain = outputs_remain['preds']
                else:
                    logits_remain = outputs_remain
                    preds_remain = outputs_remain
                
                loss_remain = loss_fn(preds_remain.squeeze(), y_remain)
                grad_remain = torch.autograd.grad(loss_remain, model.parameters(), create_graph=False, retain_graph=True)
            
            # 3. Compute fairness gradient
            fair_loss, fair_stats = compute_equalized_odds_loss(
                logits_remain.squeeze(), 
                y_remain, 
                protected_attr
            )
            
            if fair_loss.item() != 0:
                grad_fair = torch.autograd.grad(fair_loss, model.parameters(), create_graph=False)
            else:
                grad_fair = [torch.zeros_like(param) for param in model.parameters()]
            
            # 4. Apply orthogonalization
            grad_forget_processed = gram_schmidt_process(grad_forget, [grad_remain, grad_fair])
            
            # 5. Apply combined gradient update
            unlearning_lr = config.unlearning_lr
            if step > 5:
                unlearning_lr *= 0.8
            
            effective_weight_forget = config.weight_cls / len(remaining_loader)
            
            with torch.no_grad():
                for param, g_forget, g_remain, g_fair in zip(model.parameters(), grad_forget_processed, grad_remain, grad_fair):
                    final_grad = (effective_weight_forget * g_forget + 
                                config.weight_remain * g_remain + 
                                config.weight_fair * g_fair)
                    param.data -= unlearning_lr * final_grad
            
            # Update statistics
            total_stats['loss_forget'] += loss_forget.item()
            total_stats['loss_remain'] += loss_remain.item()
            total_stats['fair_loss'] += fair_loss.item() if fair_loss.item() != 0 else 0.0
            batch_count += 1
            
            # Memory cleanup
            torch.cuda.empty_cache()
        
        total_stats['batch_count'] = batch_count
    
    # Average the stats
    if total_stats['batch_count'] > 0:
        total_stats['loss_forget'] /= total_stats['batch_count']
        total_stats['loss_remain'] /= total_stats['batch_count']
        total_stats['fair_loss'] /= total_stats['batch_count']
    
    return total_stats
