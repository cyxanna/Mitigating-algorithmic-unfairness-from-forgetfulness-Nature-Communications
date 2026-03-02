import copy
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from evaluation.eval import evaluate_cls_fairness


class EarlyStopping:
    def __init__(self, *, min_delta=0.0, patience=0):
        self.min_delta = min_delta
        self.patience = patience
        self.best = float("inf")
        self.wait = 0
        self.done = False

    def step(self, current):
        self.wait += 1

        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        elif self.wait >= self.patience:
            self.done = True

        return self.done


def train_original_model(model, train_loader, val_dataset, loss_fn, model_save_path, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.to(config.device)
    best_val_loss = float('inf')
    verbose = False
    best_model = None
    best_val_auc = 0

    verboseprint = print if verbose else lambda *a, **k: None
    early_stopping = EarlyStopping(patience=config.early_stop_patience)
    real_epoch = config.epochs

    for epoch in range(0, config.epochs):
        train_loss = 0
        for i, (inputs, label, protected_attr) in enumerate(train_loader):
            inputs = inputs.to(config.device)
            label = label.to(config.device)
            protected_attr = protected_attr.to(config.device)
            model.train()
            optimizer.zero_grad()
            output = model(inputs)
            if isinstance(output, dict):
                pred = output['preds'].squeeze()
            else:
                pred = output.squeeze()
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()

            train_loss = loss.detach().cpu().item()

        val_metrics, val_loss, original_threshold = evaluate_cls_fairness(model, val_dataset, config.device, eval_fairness=False, protected_attr=None)
        
        print(f'Epoch [{epoch + 1}/{config.epochs}], Test AUROC: {val_metrics["au-roc"]:.4f}')

        if val_metrics["au-roc"] > best_val_auc:
            best_val_auc = val_metrics["au-roc"]
            best_model = copy.deepcopy(model)

        if early_stopping.step(val_loss):
            print("Early stopping .")
            real_epoch = epoch + 1
            break

    print(f"Final test AUROC: {val_metrics['au-roc']:.4f}")
    if model_save_path is not None:
        print(f"Best checkpoint saved ...")
        torch.save(best_model.state_dict(), model_save_path)
    else:
        print(f"Best checkpoint not saved ...")

    return best_model if best_model is not None else model, original_threshold, real_epoch
