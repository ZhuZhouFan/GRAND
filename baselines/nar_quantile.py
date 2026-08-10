"""
Train NAR quantile (or mean) models on the empirical sample, with a frozen SGA graph.

Inputs: feature/label tensors under ``{project_path}/tensor/lag_{S}_horizon_{h}/``
and a pretrained SGA checkpoint (best ``K`` for ``--tau``); requires ``--tau``,
``--lr``, ``--hidden``, ``--lag``, and ``--cuda``. Use ``--mse-loss`` with
``--tau 0.0`` for the conditional-mean model.
Operations: freeze the SGA graph encoder, estimate the linear NAR on the
combined training/validation window, and early-stop on that same sample.
Outputs: ``network_best.pth`` under
``{project_path}/nar_quantile_model/hidden_*_lr_*_lag_*_horizon_*/{tau}/``.
"""
import torch
import pandas as pd
import argparse

import sys
sys.path.append('.')
from config import project_path, start_time, valid_time, horizon, lag
from network.model import SGA
from network.NAR_agent import NAR_agent
from empirical_analysis.train_quantile import fetch_basic_attributes


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
                        help='Use the MSE as the loss (i.e., mean model)')
    parser.add_argument('--tau', type=float,
                        help='Quantile level')
    parser.add_argument('--lam', type=float, default=0,
                        help='Tuning parameter for L1 loss.')

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

    tensor_path = f'{project_path}/tensor/lag_{lag_order}_horizon_{horizon}'
    graph_model_path = f'{project_path}/quantile_model/hidden_{args.hidden}_lr_{args.lr}_lag_{lag_order}_horizon_{horizon}/{tau}'
    log_dir = f'{project_path}/nar_quantile_model/hidden_{args.hidden}_lr_{args.lr}_lag_{lag_order}_horizon_{horizon}/{tau}'

    N, P, _ = fetch_basic_attributes(tensor_path, start_time, valid_time)

    K_df = pd.read_csv(f'{graph_model_path}/K_result.csv')
    K = K_df.loc[K_df['best_score'].argmin(), 'K']
    graph_model = SGA(individual_num=N,
                      feature_dim=P,
                      K=K,
                      hidden_dim=args.hidden,
                      num_layers=2,
                      dropout=args.dropout)
    graph_model.load_state_dict(torch.load(f'{graph_model_path}/{K}/network_best.pth',
                                           map_location=graph_model.device))

    agent = NAR_agent(individual_num=N,
                      feature_dim=P,
                      graph_model=graph_model,
                      ret_col=args.ret_col,
                      log_dir=log_dir,
                      patience=args.patience,
                      learning_rate=args.lr,
                      seed=args.seed,
                      )
    # Paper: estimate NAR on the combined training and validation samples.
    agent.load_data(tensor_path, start_time, valid_time, num_workers)
    agent.train(tau=tau, epoch=args.epochs, lambda_=args.lam, mse_loss=args.mse_loss)
