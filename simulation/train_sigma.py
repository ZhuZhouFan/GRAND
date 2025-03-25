import torch
import pandas as pd
import argparse
import sys
sys.path.append('.')

from config import project_path
from network.sigma_agent import SGA_agent

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ## arguments related to training ##
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, 
                        help='Learning rate.')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--workers', type=int, default=3,
                        help='Number of workers in Dataloader')
    parser.add_argument('--lag', type=int, default=48,
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
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units in encoder.')

    ## Saving, loading etc. ##
    parser.add_argument('--cuda', type=int,
                        help='Number of GPU device training on.')
    parser.add_argument('--N', type=int, default=2000, help='Number of assets in the simulation')
    parser.add_argument('--T', type=int, default=600, help='Number of time periods for the simulation')
    parser.add_argument('--K', type=int, default=3, help='Number of top nodes used for adjacency matrix construction')
    parser.add_argument('--dseed', type=int, default=0, help='Random seed for DGP')
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)
    tau = args.tau
    num_workers = args.workers
    lag_order = args.lag
    N = args.N
    T = args.T
    K_star = args.K
    data_seed = args.dseed
    
    sigma_tensor_path = f'{project_path}/simulated_sigma_tensor/{N}_{T}_{K_star}_{data_seed}/hidden_{args.hidden}_lr_{args.lr}_lag_{lag_order}_horizon_1'
    
    log_dir = f'{project_path}/simulated_sigma_model/{N}_{T}_{K_star}_{data_seed}/hidden_{args.hidden}_lr_{args.lr}_lag_{lag_order}_horizon_1'
    
    K_list = [K_star]
    result_df = pd.DataFrame(columns=['K', 'best_score', 'best_cr'])
    result_df['K'] = K_list
    result_df.set_index('K', inplace=True)

    for K in K_list:
        log_dir_ = f'{log_dir}/{tau}/{K}'
        agent = SGA_agent(individual_num=N,
                           feature_dim=4+4,
                           hidden_dim=args.hidden,
                           K=K,
                           log_dir=log_dir_,
                           learning_rate=args.lr,
                           seed=args.seed,
                           )
        agent.load_data(sigma_tensor_path, str(0), str(200), str(400), num_workers)
        agent.train(tau=tau, epoch=args.epochs, lambda_=args.lam, mse_loss=args.mse_loss)

        result_df.loc[K, 'best_score'] = agent.best_score
        result_df.loc[K, 'best_cr'] = agent.best_cr
        result_df.to_csv(f'{log_dir}/{tau}/K_result.csv', index=True)