import pandas as pd
import numpy as np
import os
import argparse
import sys
sys.path.append('.')

from config import project_path

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--N', type=int, default=2000, help='Number of assets in the simulation')
    parser.add_argument('--T', type=int, default=600, help='Number of time periods for the simulation')
    parser.add_argument('--K', type=int, default=3, help='Number of top nodes used for adjacency matrix construction')
    parser.add_argument('--dseed', type=int, default=0, help='Random seed for DGP')
    
    parser.add_argument('--lr', type=float, default=5e-4, 
                        help='Learning rate.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units in encoder.')
    parser.add_argument('--lag', type=int, default=48,
                        help='Number of lagged value of each feature')
    
    args = parser.parse_args()
    
    moments_tensor = np.zeros([args.N, args.T, 4])
    moment_path = f'{project_path}/simulated_moment/{args.N}_{args.T}_{args.K}_{args.dseed}/hidden_{args.hidden}_lr_{args.lr}_lag_{args.lag}_horizon_1'
    tensor_path = f'{project_path}/simulated_tensor/{args.N}_{args.T}_{args.K}_{args.dseed}'
    sigma_tensor_path = f'{project_path}/simulated_sigma_tensor/{args.N}_{args.T}_{args.K}_{args.dseed}/hidden_{args.hidden}_lr_{args.lr}_lag_{args.lag}_horizon_1'
    
    for stock_id in range(args.N):
        stock_moment_df = pd.read_csv(f'{moment_path}/stock_{stock_id}.csv', index_col = 'date')
        moments_tensor[stock_id, :, :] = stock_moment_df[['mean', 'variance', 'skewness', 'kurtosis']].values
        
    for date in range(args.lag, args.T):
        feature = np.load(f'{tensor_path}/{date}/feature.npy')
        est_moments = moments_tensor[:, date - args.lag:date, :]
        sigma_feature = np.concatenate([feature, est_moments], axis = -1)
        sigma_label = moments_tensor[:, date, 1:2]

        os.makedirs(f'{sigma_tensor_path}/{date}', exist_ok=True)
        np.save(f'{sigma_tensor_path}/{date}/feature.npy', sigma_feature)
        np.save(f'{sigma_tensor_path}/{date}/label.npy', sigma_label)