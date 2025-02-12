from .build_model import build_model
from .modified_args import ModifiedArgs
from .get_transform import get_transform
import argparse
import sys
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import nni
import os
import random
from typing import List
import logging
import pdb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.transforms import Compose
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.dataset import VascularDataset
from src.dataset.maskable_list import MaskableList
from src.utils import get_y
from .train_and_evaluate import train_and_evaluate, evaluate


def seed_everything(seed):
    print(f"seed for seed_everything(): {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # set random seed for numpy
    torch.manual_seed(seed)  # set random seed for CPU
    torch.cuda.manual_seed_all(seed)  # set random seed for all GPUs
    torch.backends.cudnn.deterministic = True  # for reproducibility


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    if args.enable_nni:
        args = ModifiedArgs(args, nni.get_next_parameter())

    # init model
    model_name = str(args.model_name).lower()
    seed_everything(args.seed)
    device = torch.device(f'cuda:{args.device_number}' if torch.cuda.is_available() else 'cpu')
    self_dir = os.path.dirname(os.path.realpath(__file__))

    # Set up dataset paths
    root_dir = os.path.join(self_dir, 'datasets/BrainVasculature')
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        dataset_name = 'BrainVasculature_Gender'
        
    dataset = VascularDataset(root=root_dir,
                           name=dataset_name,
                           pre_transform=get_transform(args.node_features) if args.node_features is not None else None)   
    
    # Log original age values without normalization
    y_values = np.array([data.y.item() for data in dataset])
    logging.info(f"\nOriginal target values:")
    logging.info(f"Range: [{y_values.min():.2f}, {y_values.max():.2f}]")
    logging.info(f"Mean: {np.mean(y_values):.2f}")
    logging.info(f"Std: {np.std(y_values):.2f}")
    
    # Store original values for verification
    dataset.original_values = y_values.copy()
    
    y = get_y(dataset)
    num_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    nodes_num = dataset.num_nodes

    mses, maes, r2s = [], [], []
    best_mse, best_mae, best_r2 = float('inf'), float('inf'), float('-inf')
    best_mse_model, best_mae_model, best_r2_model = None, None, None

    for _ in range(args.repeat):
        # Use fixed random state for reproducible splits
        kf = KFold(n_splits=args.k_fold_splits, shuffle=True, random_state=args.seed)
        for fold_idx, (train_index, test_index) in enumerate(kf.split(dataset)):
            exp_name = f'{args.dataset_name}_{args.model_name}_fold_{fold_idx}'
            
            # Get train/test sets before normalization
            train_set = [data for data in dataset[train_index]]  # Convert to list of Data objects
            test_set = [data for data in dataset[test_index]]    # Convert to list of Data objects
            
            # Store original values before normalization
            train_y_values = np.array([data.y.item() for data in train_set])
            test_y_values = np.array([data.y.item() for data in test_set])
            
            # Calculate min-max normalization parameters using only training data
            train_y_min = np.min(train_y_values)
            train_y_max = np.max(train_y_values)
            
            logging.info(f"\nFold {fold_idx} - Training set statistics before normalization:")
            logging.info(f"Min: {train_y_min:.2f}, Max: {train_y_max:.2f}")
            logging.info(f"Mean: {np.mean(train_y_values):.2f}")
            logging.info(f"Std: {np.std(train_y_values):.2f}")
            
            # Create deep copies and normalize the data
            normalized_train_set = []
            normalized_test_set = []
            
            for data in train_set:
                data_copy = data.clone()
                data_copy.y_original = data_copy.y.clone()
                normalized_value = (data_copy.y.item() - train_y_min) / (train_y_max - train_y_min)
                data_copy.y = torch.tensor([normalized_value], dtype=torch.float)
                normalized_train_set.append(data_copy)
            
            for data in test_set:
                data_copy = data.clone()
                data_copy.y_original = data_copy.y.clone()
                normalized_value = (data_copy.y.item() - train_y_min) / (train_y_max - train_y_min)
                data_copy.y = torch.tensor([normalized_value], dtype=torch.float)
                normalized_test_set.append(data_copy)
            
            # Create new MaskableList objects with the normalized data
            train_set = MaskableList(normalized_train_set)
            test_set = MaskableList(normalized_test_set)
            
            # Add denormalization method to both sets
            def create_denormalize(min_val, max_val):
                def denormalize(normalized_values):
                    if isinstance(normalized_values, torch.Tensor):
                        normalized_values = normalized_values.numpy()
                    return normalized_values * (max_val - min_val) + min_val
                return denormalize
            
            train_set.denormalize = create_denormalize(train_y_min, train_y_max)
            test_set.denormalize = create_denormalize(train_y_min, train_y_max)
            
            # Store normalization parameters
            train_set.y_min = train_y_min
            train_set.y_max = train_y_max
            test_set.y_min = train_y_min
            test_set.y_max = train_y_max
            
            # Store original values
            train_set.original_values = train_y_values
            test_set.original_values = test_y_values
            
            # Log normalized values statistics
            train_normalized = np.array([data.y.item() for data in train_set])
            test_normalized = np.array([data.y.item() for data in test_set])
            logging.info(f"\nNormalized values statistics:")
            logging.info(f"Train - Range: [{train_normalized.min():.4f}, {train_normalized.max():.4f}]")
            logging.info(f"Test - Range: [{test_normalized.min():.4f}, {test_normalized.max():.4f}]")
            
            # Verify normalization
            logging.info(f"\nVerification:")
            logging.info(f"Original train range: [{train_y_values.min():.2f}, {train_y_values.max():.2f}]")
            logging.info(f"Original test range: [{test_y_values.min():.2f}, {test_y_values.max():.2f}]")
            denormalized_train = train_set.denormalize(train_normalized)
            denormalized_test = test_set.denormalize(test_normalized)
            logging.info(f"Denormalized train range: [{denormalized_train.min():.2f}, {denormalized_train.max():.2f}]")
            logging.info(f"Denormalized test range: [{denormalized_test.min():.2f}, {denormalized_test.max():.2f}]")
            
            # Create data loaders
            train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)
            
            # Build model and continue with training
            model = build_model(args, device, model_name, num_features, nodes_num, num_edge_features=num_edge_features, n_classes=1)
            
            # Use AdamW optimizer with decoupled weight decay
            if args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            # train
            test_mse, test_mae, test_r2 = train_and_evaluate(model, train_loader, test_loader,
                                                          optimizer, device, args)

            # Get metrics from evaluate function
            test_metrics = evaluate(model, device, test_loader)
            if hasattr(model, 'regression') and model.regression:
                test_mse = test_metrics["mse"]
                test_mae = test_metrics["mae"]
                test_r2 = test_metrics["r2"]
            else:
                test_micro, test_auc, test_macro = test_metrics
                test_mse, test_mae, test_r2 = test_micro, test_auc, test_macro

            logging.info(f'(Initial Performance Last Epoch) | test_mse={test_mse:.4f}, '
                         f'test_mae={test_mae:.4f}, test_r2={test_r2:.4f}')

            mses.append(test_mse)
            maes.append(test_mae)
            r2s.append(test_r2)

            # Save the best models based on each metric
            if test_mse < best_mse:
                best_mse = test_mse
                best_mse_model = model.state_dict()
            if test_mae < best_mae:
                best_mae = test_mae
                best_mae_model = model.state_dict()
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_r2_model = model.state_dict()

    if args.enable_nni:
        nni.report_final_result({
            'default': np.mean(r2s),
            'r2': np.mean(r2s),
            'mse': np.mean(mses),
            'mae': np.mean(maes)
        })

    ## create a log folder under exp/ and save the model
    os.makedirs('exp/log', exist_ok=True)
    exp_name = f'{args.dataset_name}_{args.model_name}_{args.node_features}_{args.pooling}_{args.gcn_mp_type}_{args.gat_mp_type}_{args.num_heads}_{args.hidden_dim}_{args.gat_hidden_dim}_{args.edge_emb_dim}_{args.lr}_{args.weight_decay}_{args.dropout}_{args.seed}_{args.mixup}'   
    os.makedirs(f'exp/log/{exp_name}', exist_ok=True)
    
    # save models
    torch.save(model.state_dict(), f'exp/log/{exp_name}/model.pth')
    torch.save(best_mse_model, f'exp/log/{exp_name}/best_mse_model.pth')
    torch.save(best_mae_model, f'exp/log/{exp_name}/best_mae_model.pth')
    torch.save(best_r2_model, f'exp/log/{exp_name}/best_r2_model.pth')    
    
    # plot the test results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.plot(mses, label='MSE')
    plt.title('Mean Squared Error')
    plt.xlabel('Fold')
    plt.ylabel('MSE')
    
    plt.subplot(132)
    plt.plot(maes, label='MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Fold')
    plt.ylabel('MAE')
    
    plt.subplot(133)
    plt.plot(r2s, label='R²')
    plt.title('R² Score')
    plt.xlabel('Fold')
    plt.ylabel('R²')
    
    plt.tight_layout()
    plt.savefig(f'exp/log/{exp_name}/test_results.png')
    plt.close()

    result_str = f'(K Fold Final Result)| avg_mse={np.mean(mses):.4f} +- {np.std(mses):.4f}, ' \
                 f'avg_mae={np.mean(maes):.4f} +- {np.std(maes):.4f}, ' \
                 f'avg_r2={np.mean(r2s):.4f} +- {np.std(r2s):.4f}\n'
    logging.info(result_str)

    print("seed for main(): ", args.seed)

    with open('result.log', 'a') as f:
        input_arguments: List[str] = sys.argv
        f.write(f'{input_arguments}\n')
        f.write(result_str + '\n')
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler('result.log', 'a')])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str,
                        default="BrainVasculature_AGE")
    parser.add_argument('--view', type=int, default=1)
    parser.add_argument('--node_features', type=str,
                        choices=['identity', 'degree', 'degree_bin', 'LDP', 'node2vec', 'adj', 'diff_matrix',
                                 'eigenvector', 'eigen_norm', 'adj_more'],
                        default=None)
    parser.add_argument('--pooling', type=str,
                        choices=['sum', 'concat', 'mean'],
                        default='mean')
                        
    parser.add_argument('--model_name', type=str, default='gat')
    parser.add_argument('--gcn_mp_type', type=str, default="node_concate") 
    parser.add_argument('--gat_mp_type', type=str, default="attention_edge_weighted") 

    parser.add_argument('--enable_nni', action='store_true')
    parser.add_argument('--n_GNN_layers', type=int, default=2)
    parser.add_argument('--n_MLP_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--gat_hidden_dim', type=int, default=4)
    parser.add_argument('--edge_emb_dim', type=int, default=64)
    parser.add_argument('--edge_attr_size', type=int, default=2)
    parser.add_argument('--bucket_sz', type=float, default=0.05)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--k_fold_splits', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--test_interval', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=1)

    parser.add_argument('--seed', type=int, default=1301)
    parser.add_argument('--diff', type=float, default=0.2)
    parser.add_argument('--mixup', type=int, default=0)
    parser.add_argument('--device_number', type=int, default=0)

    parser.add_argument('--regression', action='store_true')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw'], default='adamw')
    parser.add_argument('--lr_schedule', type=str, choices=['none', 'cosine', 'onecycle'], default='cosine')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--use_huber', action='store_true')
    parser.add_argument('--huber_delta', type=float, default=1.0)
    parser.add_argument('--clip_grad', action='store_true')
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()
    main(args)
