import torch
import pandas as pd
import numpy as np
import argparse
import os

from node_fuse_agent import mlp_agent


def fetch_basic_attributes(tensor_path,
                           start_time,
                           end_time):

    dates = np.array(os.listdir(tensor_path))
    avaible_dates = dates[(dates >= start_time) & (dates <= end_time)]
    avaible_dates.sort()
    valid_time = avaible_dates[round(len(avaible_dates) * 0.8)]

    feature = np.load(f'{tensor_path}/{dates[0]}/feature.npy')
    N = feature.shape[0]
    P = feature.shape[-1]

    return N, P, valid_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ## arguments related to training ##
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--workers', type=int, default=3,
                        help='Number of workers in Dataloader')
    parser.add_argument('--lag', type=int,
                        help='Number of lagged value of each feature')

    ## arguments related to loss function ##
    parser.add_argument('--mse-loss', action='store_true', default=False,
                        help='Use the MSE as the loss (i.e., mean model of quantile model)')
    parser.add_argument('--tau', type=float,
                        help='Quantile level')
    parser.add_argument('--lam', type=float, default=0,
                        help='Tuning parameter for L1 loss.')

    ## arguments related to weight and bias initialisation ##
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')

    ## arguments related to changing the model ##
    parser.add_argument('--hidden', type=int,
                        help='Number of hidden units in encoder.')

    ## Saving, loading etc. ##
    parser.add_argument('--cuda', type=int,
                        help='Number of GPU device training on.')
    parser.add_argument('--data-folder', type=str,
                        help='Path to your data folder')
    parser.add_argument('--start-time', type=str, default='2010-01-01',
                        help='Beginning of in-sample dataset')
    parser.add_argument('--end-time', type=str, default='2018-12-31',
                        help='Ending of in-sample dataset')

    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)
    start_time = args.start_time
    end_time = args.end_time
    tau = args.tau
    num_workers = args.workers
    data_path = args.data_folder
    lag_order = args.lag

    tensor_path = f'{data_path}/tensor/lag_{lag_order}_horizon_1'
    quantile_model_path = f'{data_path}/quantile_model/hidden_{args.hidden}_lr_{args.lr}_lag_{lag_order}_horizon_1'

    N, P, valid_time = fetch_basic_attributes(
        tensor_path, start_time, end_time)
    log_dir = f'{data_path}/mlp_model/hidden_{args.hidden}_lr_{args.lr}_lag_{lag_order}_horizon_1/{tau}'

    K_df = pd.read_csv(f'{quantile_model_path}/{tau}/K_result.csv')
    K = K_df.loc[K_df['best_score'].argmin(), 'K']

    agent = mlp_agent(individual_num=N,
                      feature_dim=P,
                      hidden_dim=args.hidden,
                      log_dir=log_dir,
                      K=K,
                      learning_rate=args.lr,
                      seed=args.seed,
                      )
    agent.load_data(tensor_path, start_time,
                    valid_time, end_time, num_workers)
    agent.train(tau=tau, epoch=args.epochs,
                lambda_=args.lam, mse_loss=args.mse_loss)
