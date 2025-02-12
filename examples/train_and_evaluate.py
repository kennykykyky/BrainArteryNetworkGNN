import numpy as np
import nni
import torch
import pdb
import torch.nn.functional as F
from sklearn import metrics
from typing import Optional
from torch.utils.data import DataLoader
import logging
import os
from src.utils import mixup, mixup_criterion
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import matplotlib.pyplot as plt


def setup_denorm_logger(exp_name):
    """Set up a dedicated logger for denormalization verification"""
    os.makedirs('exp/denorm_logs', exist_ok=True)
    denorm_logger = logging.getLogger('denormalization')
    denorm_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicate logging
    if denorm_logger.hasHandlers():
        denorm_logger.handlers.clear()
    
    # Create file handler
    fh = logging.FileHandler(f'exp/denorm_logs/{exp_name}_denorm.log')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    denorm_logger.addHandler(fh)
    return denorm_logger


def train_and_evaluate(model, train_loader, test_loader, optimizer, device, args):
    model.train()
    accs, aucs, macros = [], [], []
    epoch_num = args.epochs
    
    # Set up denormalization logger
    exp_name = f"{args.dataset_name}_{args.model_name}"
    denorm_logger = setup_denorm_logger(exp_name)
    
    # Initialize learning rate scheduler
    if args.lr_schedule == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=args.min_lr)
    elif args.lr_schedule == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=epoch_num, 
                             steps_per_epoch=len(train_loader),
                             pct_start=0.1)  # 10% warmup
    
    # Initialize Huber loss if specified
    huber_loss = torch.nn.HuberLoss(delta=args.huber_delta) if args.use_huber else None
    
    # Track losses for analysis
    train_losses = []
    test_losses = []
    train_metrics_history = []
    test_metrics_history = []
    
    for i in range(epoch_num):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            if hasattr(model, 'regression') and model.regression:
                pred = model(data)
                
                if hasattr(train_loader.dataset, 'use_bce') and train_loader.dataset.use_bce:
                    loss = F.binary_cross_entropy_with_logits(pred.view(-1), data.y.float())
                elif args.use_huber:
                    loss = huber_loss(pred.view(-1), data.y.float())
                else:
                    loss = F.mse_loss(pred.view(-1), data.y.float())
            else:
                if args.mixup:
                    data, y_a, y_b, lam = mixup(data, device=device)
                out = model(data)
                if args.mixup:
                    loss = mixup_criterion(F.nll_loss, out, y_a, y_b, lam)
                else:
                    loss = F.nll_loss(out, data.y)
            
            loss.backward()
            
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
                
            optimizer.step()
            
            if args.lr_schedule == 'onecycle':
                scheduler.step()
                
            epoch_loss += loss.item()
            num_batches += 1
        
        # Average training loss for the epoch
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Step the learning rate scheduler if using cosine
        if args.lr_schedule == 'cosine':
            scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log training metrics
        model.eval()
        with torch.no_grad():
            train_metrics = evaluate(model, device, train_loader, i, denorm_logger)
            if hasattr(model, 'regression') and model.regression:
                logging.info(
                    f'Epoch {i:03d} (Train) | '
                    f'Loss: {avg_train_loss:.4f} | '
                    f'R²: {train_metrics["r2"]:.4f} | '
                    f'MSE: {train_metrics["mse"]:.4f} | '
                    f'MAE: {train_metrics["mae"]:.4f} | '
                    f'LR: {current_lr:.6f}'
                )
            else:
                logging.info(
                    f'Epoch {i:03d} (Train) | '
                    f'Loss: {avg_train_loss:.4f} | '
                    f'Micro-F1: {(train_metrics[0] * 100):.2f} | '
                    f'Macro-F1: {(train_metrics[2] * 100):.2f} | '
                    f'AUC: {(train_metrics[1] * 100):.2f} | '
                    f'LR: {current_lr:.6f}'
                )

        # Evaluate test metrics
        if (i + 1) % args.test_interval == 0:
            model.eval()
            with torch.no_grad():
                test_metrics = evaluate(model, device, test_loader, i, denorm_logger)
                
                if hasattr(model, 'regression') and model.regression:
                    train_metrics_history.append(train_metrics)
                    test_metrics_history.append(test_metrics)
                    
                    # Compute test loss
                    test_loss = 0
                    num_test_batches = 0
                    for data in test_loader:
                        data = data.to(device)
                        pred = model(data)
                        if args.use_huber:
                            test_loss += huber_loss(pred.view(-1), data.y.float()).item()
                        else:
                            test_loss += F.mse_loss(pred.view(-1), data.y.float()).item()
                        num_test_batches += 1
                    avg_test_loss = test_loss / num_test_batches
                    test_losses.append(avg_test_loss)
                    
                    # Log test metrics
                    logging.info(
                        f'Epoch {i:03d} (Test) | '
                        f'Loss: {avg_test_loss:.4f} | '
                        f'R²: {test_metrics["r2"]:.4f} | '
                        f'MSE: {test_metrics["mse"]:.4f} | '
                        f'MAE: {test_metrics["mae"]:.4f}'
                    )
                else:
                    # Compute test loss for classification
                    test_loss = 0
                    num_test_batches = 0
                    for data in test_loader:
                        data = data.to(device)
                        out = model(data)
                        test_loss += F.nll_loss(out, data.y).item()
                        num_test_batches += 1
                    avg_test_loss = test_loss / num_test_batches
                    test_losses.append(avg_test_loss)
                    
                    # Log test metrics
                    logging.info(
                        f'Epoch {i:03d} (Test) | '
                        f'Loss: {avg_test_loss:.4f} | '
                        f'Micro-F1: {(test_metrics[0] * 100):.2f} | '
                        f'Macro-F1: {(test_metrics[2] * 100):.2f} | '
                        f'AUC: {(test_metrics[1] * 100):.2f}'
                    )
    
    if hasattr(model, 'regression') and model.regression:
        return test_metrics["mse"], test_metrics["mae"], test_metrics["r2"]
    else:
        return np.mean(accs), np.mean(aucs), np.mean(macros)


@torch.no_grad()
def evaluate(model, device, loader, epoch=None, denorm_logger=None):
    model.eval()
    
    if hasattr(model, 'regression') and model.regression:
        # Soft classification evaluation
        predictions = []
        targets = []
        for data in loader:
            data = data.to(device)
            pred = model(data)
            # For regression, we don't need sigmoid
            predictions.append(pred.cpu())
            targets.append(data.y.float().cpu())
        
        predictions = torch.cat(predictions, dim=0).numpy().flatten()
        targets = torch.cat(targets, dim=0).numpy().flatten()
        
        # Denormalize predictions and targets if dataset has normalization parameters
        if hasattr(loader.dataset, 'denormalize'):
            # Store normalized values for verification
            normalized_predictions = predictions.copy()
            normalized_targets = targets.copy()
            
            # Denormalize
            predictions = loader.dataset.denormalize(predictions)
            targets = loader.dataset.denormalize(targets)
            
        return {
            "mse": metrics.mean_squared_error(targets, predictions),
            "mae": metrics.mean_absolute_error(targets, predictions),
            "r2": metrics.r2_score(targets, predictions)
        }
    else:
        # Original classification evaluation
        preds, trues, preds_prob = [], [], []
        
        for data in loader:
            data = data.to(device)
            res = model(data)
            
            # Check the number of classes
            n_classes = res.shape[1]  # Output dimension (number of classes)

            # Get predictions
            if n_classes == 2:  # Binary classification
                # Extract probabilities using softmax (class 1 probability)
                probs = torch.softmax(res, dim=1)[:, 1]  # Class 1 probability
                preds += (probs > 0.5).long().detach().cpu().tolist()  # Threshold at 0.5
                preds_prob += probs.detach().cpu().tolist()
            else:  # Multi-class classification
                # Apply softmax to get probabilities for all classes
                probs = torch.softmax(res, dim=1)
                preds += torch.argmax(probs, dim=1).detach().cpu().tolist()  # Predicted class
                preds_prob += probs.detach().cpu().tolist()  # Full probability matrix

            trues += data.y.detach().cpu().tolist()

        # check whether the model is binary classification
        if np.unique(trues).shape[0] == 2:
            auc = metrics.roc_auc_score(trues, preds_prob)
        else:
            auc = metrics.roc_auc_score(trues, preds_prob, multi_class='ovr')

        if np.isnan(auc):
            train_auc = 0.5
        else:
            train_auc = auc
            
        train_micro = metrics.f1_score(trues, preds, average='micro')
        if np.unique(trues).shape[0] == 2:
            train_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])
        else:
            train_macro = metrics.f1_score(trues, preds, average='macro')

        return train_micro, train_auc, train_macro
