"""
Train the NAR node-fusion MLP for a given quantile level (empirical setting).

Inputs: quantile tensors under ``{project_path}/tensor/...``, a trained SGA
checkpoint (best ``K``), and a trained NAR checkpoint for ``--tau``; use
matching ``--lr`` / ``--hidden`` / ``--lag``.
Operations: randomly fuse pairs of nodes through an MLP, build the graph with
the frozen SGA encoder, evaluate the fused cross-section with the frozen NAR,
and optimize the fusion parameters.
Outputs: ``mlp_best.pth`` / ``mlp_final.pth`` under
``{project_path}/nar_mlp_model/hidden_*_lr_*_lag_*_horizon_{h}/{tau}/``.
"""

import torch
import pandas as pd
import argparse

import sys
sys.path.append('.')

from config import project_path, start_time, valid_time, horizon, lag
from network.NAR_node_fuse_agent import NAR_mlp_agent
from empirical_analysis.train_mlp import fetch_basic_attributes


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ## arguments related to training ##
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--workers', type=int, default=3,
                        help='Number of workers in Dataloader')
    parser.add_argument('--lag', type=int, default=lag,
                        help='Number of lagged value of each feature')

    ## arguments related to loss function ##
    parser.add_argument('--mse-loss', action='store_true', default=False,
                        help='Use the MSE as the loss (i.e., mean model of quantile model)')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='Quantile level')
    parser.add_argument('--lam', type=float, default=0,
                        help='Tuning parameter for L1 loss.')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size.')

    ## arguments related to weight and bias initialisation ##
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')

    ## arguments related to changing the model ##
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units in encoder.')
    parser.add_argument('--ret-col', type=int, default=7,
                        help='Feature column index of lagged return (log_ret).')

    ## Saving, loading etc. ##
    parser.add_argument('--cuda', type=int, default=0,
                        help='Number of GPU device training on.')
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)
    tau = args.tau
    num_workers = args.workers
    lag_order = args.lag
    hidden = args.hidden
    lr = args.lr

    tensor_path = f'{project_path}/tensor/lag_{lag_order}_horizon_{horizon}'
    graph_model_path = f'{project_path}/quantile_model/hidden_{hidden}_lr_{lr}_lag_{lag_order}_horizon_{horizon}/{tau}'
    nar_model_path = f'{project_path}/nar_quantile_model/hidden_{hidden}_lr_{lr}_lag_{lag_order}_horizon_{horizon}/{tau}'
    log_dir = f'{project_path}/nar_mlp_model/hidden_{hidden}_lr_{lr}_lag_{lag_order}_horizon_{horizon}/{tau}'

    N, P, split_point = fetch_basic_attributes(tensor_path, start_time, valid_time)

    K_df = pd.read_csv(f'{graph_model_path}/K_result.csv')
    K = K_df.loc[K_df['best_score'].argmin(), 'K']

    agent = NAR_mlp_agent(individual_num=N,
                          feature_dim=P,
                          hidden_dim=hidden,
                          log_dir=log_dir,
                          graph_model_path=graph_model_path,
                          NAR_model_path=nar_model_path,
                          K=K,
                          ret_col=args.ret_col,
                          learning_rate=args.lr,
                          seed=args.seed,
                          batch_size=args.batch,
                          patience=args.patience,
                          dropout=args.dropout,
                          )
    agent.load_data(tensor_path,
                    start_time,
                    split_point,
                    valid_time,
                    num_workers)
    agent.train(tau=tau,
                epoch=args.epochs,
                lambda_=args.lam,
                mse_loss=args.mse_loss)
