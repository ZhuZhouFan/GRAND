"""
Train the SGA-based sigma (conditional variance) model.

Inputs: sigma feature/label tensors under
``{project_path}/sigma_tensor/hidden_*_lr_*_lag_*_horizon_{h}/``; requires
``--lr``, ``--hidden``, ``--lag``, and ``--cuda`` matching the moment / sigma
tensor directory names.
Operations: MSE training over ``K`` in 1..20 with early stopping on the
in-sample validation split.
Outputs: checkpoints and ``K_result.csv`` under
``{project_path}/sigma_model/hidden_*_lr_*_lag_*_horizon_{h}/0.0/``.
"""

import torch
import pandas as pd
import argparse

import sys
sys.path.append('.')
from config import project_path, start_time, valid_time, horizon, lag
from network.sigma_agent import SGA_agent
from empirical_analysis.train_quantile import fetch_basic_attributes

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ## arguments related to training ##
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate.')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--workers', type=int, default=3,
                        help='Number of workers in Dataloader')
    parser.add_argument('--lag', type=int, default=lag,
                        help='Number of lagged value of each feature')

    ## arguments related to loss function ##
    parser.add_argument('--lam', type=float, default=0,
                        help='tuning parameter for L1 loss.')

    ## arguments related to weight and bias initialisation ##
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')

    ## arguments related to changing the model ##
    parser.add_argument('--hidden', type=int, default=128, 
                        help='Number of hidden units in encoder.')

    ## Saving, loading etc. ##
    parser.add_argument('--cuda', type=int, default=0, 
                        help='Number of GPU device training on.')
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)
    tau = 0.0
    num_workers = args.workers
    lag_order = args.lag
    hidden = args.hidden
    lr = args.lr
    
    tensor_path = f'{project_path}/sigma_tensor/hidden_{hidden}_lr_{lr}_lag_{lag_order}_horizon_{horizon}'
    N, P, split_point = fetch_basic_attributes(tensor_path, start_time, valid_time)
    log_dir = f'{project_path}/sigma_model/hidden_{args.hidden}_lr_{args.lr}_lag_{lag_order}_horizon_{horizon}'
    
    K_list = [x+1 for x in range(20)]
    result_df = pd.DataFrame(columns=['K', 'best_score', 'best_cr'])
    result_df['K'] = K_list
    result_df.set_index('K', inplace=True)

    for K in K_list:
        log_dir_ = f'{log_dir}/{tau}/{K}'
        agent = SGA_agent(individual_num=N,
                          feature_dim=P,
                          hidden_dim=hidden,
                          K=K,
                          log_dir=log_dir_,
                          learning_rate=lr,
                          seed=args.seed,
                          )
        agent.load_data(tensor_path, start_time, split_point, valid_time, num_workers)
        agent.train(tau=tau,
                    epoch=args.epochs,
                    lambda_=args.lam,
                    mse_loss=True)

        result_df.loc[K, 'best_score'] = agent.best_score
        result_df.loc[K, 'best_cr'] = agent.best_cr
        result_df.to_csv(f'{log_dir}/{tau}/K_result.csv', index=True)