import numpy as np
import nni
import torch
import pdb
import torch.nn.functional as F
from sklearn import metrics
from typing import Optional
from torch.utils.data import DataLoader
import logging
from src.utils import mixup, mixup_criterion


def train_and_evaluate(model, train_loader, test_loader, optimizer, device, args):
    model.train()
    accs, aucs, macros = [], [], []
    epoch_num = args.epochs

    for i in range(epoch_num):
        loss_all = 0
        for data in train_loader:
            data = data.to(device)

            if args.mixup:
                data, y_a, y_b, lam = mixup(data, device=device)
            optimizer.zero_grad()
            out = model(data)

            if args.mixup:
                loss = mixup_criterion(F.nll_loss, out, y_a, y_b, lam)
            else:
                loss = F.nll_loss(out, data.y)

            loss.backward()
            optimizer.step()

            loss_all += loss.item()
        epoch_loss = loss_all / len(train_loader.dataset)

        train_micro, train_auc, train_macro = evaluate(model, device, train_loader)
        logging.info(f'(Train) | Epoch={i:03d}, loss={epoch_loss:.4f}, '
                     f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                     f'train_auc={(train_auc * 100):.2f}')

        if (i + 1) % args.test_interval == 0:
            test_micro, test_auc, test_macro = evaluate(model, device, test_loader)
            accs.append(test_micro)
            aucs.append(test_auc)
            macros.append(test_macro)
            text = f'(Train Epoch {i}), test_micro={(test_micro * 100):.2f}, ' \
                   f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}\n'
            logging.info(text)

        if args.enable_nni:
            nni.report_intermediate_result(train_auc)

    accs, aucs, macros = np.sort(np.array(accs)), np.sort(np.array(aucs)), np.sort(np.array(macros))
    return accs.mean(), aucs.mean(), macros.mean()


@torch.no_grad()
def evaluate(model, device, loader, test_loader: Optional[DataLoader] = None) -> (float, float):
    model.eval()
    preds, trues, preds_prob = [], [], []

    correct, auc = 0, 0
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

        # pred = res.max(dim=1)[1]
        # preds += pred.detach().cpu().tolist()
        # preds_prob += torch.exp(res)[:, 1].detach().cpu().tolist()
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

    if test_loader is not None:
        test_micro, test_auc, test_macro = evaluate(model, device, test_loader)
        return train_micro, train_auc, train_macro, test_micro, test_auc, test_macro
    else:
        return train_micro, train_auc, train_macro
