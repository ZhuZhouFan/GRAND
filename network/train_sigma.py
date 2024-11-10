import torch
import pandas as pd
import numpy as np
import argparse

from sigma_agent import SGA_agent
from train import fetch_basic_attributes

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ## arguments related to training ##
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float,
                        help='Initial learning rate.')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--workers', type=int, default=3,
                        help='Number of workers in Dataloader')
    parser.add_argument('--lag', type=int, 
                        help='Number of lagged value of each feature')

    ## arguments related to loss function ##
    parser.add_argument('--lam', type=float, default=0,
                        help='tuning parameter for L1 loss.')

    ## arguments related to weight and bias initialisation ##
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')

    ## arguments related to changing the model ##
    parser.add_argument('--hidden', type=int, 
                        help='Number of hidden units in encoder.')

    ## Saving, loading etc. ##
    parser.add_argument('--cuda', type=int, 
                        help='Number of device training on.')
    parser.add_argument('--data-folder', type=str,
                         help='Path to your data folder')
    parser.add_argument('--start-time', type=str, default='2010-01-01',
                        help='Beginning of in-sample dataset')
    parser.add_argument('--end-time', type=str, default='2018-12-31',
                        help='Ending of in-sample dataset')
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)
    
    num_workers = args.workers
    lag_order = args.lag
    data_path = args.data_folder
    start_time = args.start_time
    end_time = args.end_time
    
    tensor_path = f'{data_path}/sigma_tensor/hidden_{args.hidden}_lr_{args.lr}_lag_{lag_order}_horizon_1'
    log_dir = f'{data_path}/sigma_model/hidden_{args.hidden}_lr_{args.lr}_lag_{lag_order}_horizon_1'
    
    N, P, valid_time = fetch_basic_attributes(tensor_path, start_time, end_time)
    
    tau = 0.0 # just use for label sigma model
    
    # K_list = [x+1 for x in range(20)]
    
    K_list = [x+1 for x in range(10)]
    
    result_df = pd.DataFrame(columns=['K', 'best_score', 'best_cr'])
    result_df['K'] = K_list
    result_df.set_index('K', inplace=True)

    for K in K_list:
        log_dir_ = f'{log_dir}/{tau}/{K}'
        agent = SGA_agent(individual_num=N,
                          feature_dim=P,
                          hidden_dim=args.hidden,
                          K=K,
                          log_dir=log_dir_,
                          learning_rate=args.lr,
                          seed=args.seed,
                          )
        agent.load_data(tensor_path, start_time, valid_time, end_time, num_workers)
        agent.train(tau=tau, epoch=args.epochs,
                    lambda_=args.lam, mse_loss=True)

        result_df.loc[K, 'best_score'] = agent.best_score
        result_df.loc[K, 'best_cr'] = agent.best_cr
        result_df.to_csv(f'{log_dir}/{tau}/K_result.csv', index=True)