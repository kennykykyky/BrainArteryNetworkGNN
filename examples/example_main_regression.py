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

from src.dataset import VascularDataset
from src.dataset.maskable_list import MaskableList
from src.utils import get_y
# from train_and_evaluate import train_and_evaluate, evaluate
from .train_and_evaluate import train_and_evaluate, evaluate


def seed_everything(seed):
    print(f"seed for seed_everything(): {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # set random seed for numpy
    torch.manual_seed(seed)  # set random seed for CPU
    torch.cuda.manual_seed_all(seed)  # set random seed for all GPUs


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    if args.enable_nni:
        # pdb.set_trace()
        args = ModifiedArgs(args, nni.get_next_parameter())

    # init model
    model_name = str(args.model_name).lower()
    # seed_everything(args.seed) # use args.seed for each run
    device = torch.device(f'cuda:{args.device_number}' if torch.cuda.is_available() else 'cpu')
    self_dir = os.path.dirname(os.path.realpath(__file__))

    if args.dataset_name == 'BRAVE':  # ABCD dataset puts files in a separate directory
        root_dir = os.path.join(self_dir, 'datasets/BRAVE/')
    elif args.dataset_name == 'BrainVasculature':
        root_dir = os.path.join(self_dir, 'datasets/BrainVasculature/')
    elif args.dataset_name == 'ABIDE':
        root_dir = os.path.join(self_dir, 'datasets/ABIDE/')
    else:
        root_dir = os.path.join(self_dir, 'datasets/', args.dataset_name)
        
    dataset = VascularDataset(root=root_dir,
                           name=args.dataset_name,
                           pre_transform=get_transform(args.node_features) if args.node_features is not None else None,)    
        
    y = get_y(dataset)
    num_features = dataset[0].x.shape[1]
    nodes_num = dataset.num_nodes
    # if args.model_name == 'gcn':
    #     bin_edges = calculate_bin_edges(dataset, num_bins=args.bucket_num)
    # else:
    #     bin_edges = None

    mses, maes, r2s = [], [], []
    best_mse, best_mae, best_r2 = float('inf'), float('inf'), float('-inf')
    best_mse_model, best_mae_model, best_r2_model = None, None, None

    for _ in range(args.repeat):
        seed_everything(random.randint(1, 1000000))  # use random seed for each run
        kf = KFold(n_splits=args.k_fold_splits, shuffle=True)
        for train_index, test_index in kf.split(dataset):
            model = build_model(args, device, model_name, num_features, nodes_num, n_classes=1)  # Changed to 1 output for regression
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            train_set, test_set = dataset[train_index], dataset[test_index]

            train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)

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
        # Report all metrics to NNI
        nni.report_final_result({
            'default': np.mean(r2s),  # Primary metric (R²)
            'r2': np.mean(r2s),
            'mse': np.mean(mses),
            'mae': np.mean(maes)
        })

    ## create a log folder under exp/ and save the model, as well as the training log file
    os.makedirs('exp/log', exist_ok=True)
    # name the current experiement with the current time
    exp_name = f'{args.dataset_name}_{args.model_name}_{args.node_features}_{args.pooling}_{args.gcn_mp_type}_{args.gat_mp_type}_{args.num_heads}_{args.hidden_dim}_{args.gat_hidden_dim}_{args.edge_emb_dim}_{args.lr}_{args.weight_decay}_{args.dropout}_{args.seed}_{args.mixup}'   
    os.makedirs(f'exp/log/{exp_name}', exist_ok=True)
    # save the last model
    torch.save(model.state_dict(), f'exp/log/{exp_name}/model.pth')
    # save the training log
    with open(f'exp/log/{exp_name}/training.log', 'w') as f:
        f.write(f'(K Fold Final Result)| avg_mse={np.mean(mses):.4f} +- {np.std(mses):.4f}, '
                f'avg_mae={np.mean(maes):.4f} +- {np.std(maes):.4f}, '
                f'avg_r2={np.mean(r2s):.4f} +- {np.std(r2s):.4f}\n')
    
    # Save the best models
    torch.save(best_mse_model, f'exp/log/{exp_name}/best_mse_model.pth')
    torch.save(best_mae_model, f'exp/log/{exp_name}/best_mae_model.pth')
    torch.save(best_r2_model, f'exp/log/{exp_name}/best_r2_model.pth')    
    
    # plot the test results using matplotlib
    import matplotlib.pyplot as plt
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
        # write all input arguments to f
        input_arguments: List[str] = sys.argv
        f.write(f'{input_arguments}\n')
        f.write(result_str + '\n')
        # write all logging info to f
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler('result.log', 'a')])
        

def count_degree(data: np.ndarray):  # data: (sample, node, node)
    count = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        count[i, :] = np.sum(data[:, i, :] != 0, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str,
                        # choices=['PPMI', 'HIV', 'BP', 'ABCD', 'PNC', 'ABIDE'],
                        default="BP")
    parser.add_argument('--view', type=int, default=1)
    parser.add_argument('--node_features', type=str,
                        choices=['identity', 'degree', 'degree_bin', 'LDP', 'node2vec', 'adj', 'diff_matrix',
                                 'eigenvector', 'eigen_norm', 'adj_more'],
                        default=None)
    parser.add_argument('--pooling', type=str,
                        choices=['sum', 'concat', 'mean'],
                        default='mean')
                        
    parser.add_argument('--model_name', type=str, default='gat')
    # gcn_mp_type choices: weighted_sum, bin_concate, edge_weight_concate, edge_node_concate, node_concate
    parser.add_argument('--gcn_mp_type', type=str, default="node_concate") 
    # gat_mp_type choices: attention_weighted, attention_edge_weighted, sum_attention_edge, edge_node_concate, node_concate
    parser.add_argument('--gat_mp_type', type=str, default="attention_edge_weighted") 

    parser.add_argument('--enable_nni', action='store_true')
    parser.add_argument('--n_GNN_layers', type=int, default=2)
    parser.add_argument('--n_MLP_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--gat_hidden_dim', type=int, default=32)
    parser.add_argument('--edge_emb_dim', type=int, default=128)
    parser.add_argument('--edge_attr_size', type=int, default=2)
    parser.add_argument('--bucket_sz', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--dropout', type=float, default=0.7)

    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--k_fold_splits', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--test_interval', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=1)

    parser.add_argument('--seed', type=int, default=112078)
    parser.add_argument('--diff', type=float, default=0.2)
    parser.add_argument('--mixup', type=int, default=0) #[0, 1]
    parser.add_argument('--device_number', type=int, default=0)  # Add this line

    # Add regression flag
    parser.add_argument('--regression', action='store_true', help='Whether to perform regression instead of classification')

    args = parser.parse_args()
    main(args)
