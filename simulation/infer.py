import torch
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import sys
import argparse

sys.path.append('.')
from config import project_path
from network.model import SGA
from network.QCM import compute_QCM_table

def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False

def infer(N,
          P, 
          hidden_dim,
          tau, 
          date_array,
          stock_list,
          model_path,
          tensor_path,
          graph=False):
    T = date_array.shape[0]
    K_df = pd.read_csv(f'{model_path}/{tau}/K_result.csv')
    K = K_df.loc[K_df['best_score'].argmin(), 'K']
    network = SGA(individual_num=N,
                  feature_dim=P,
                  K=K,
                  hidden_dim=hidden_dim,
                  num_layers=2)
    network.to(network.device)
    
    network.load_state_dict(torch.load(f'{model_path}/{tau}/{K}/network_best.pth',
                                       map_location=network.device,
                                       weights_only=True))
    network.eval()

    result_dict = dict.fromkeys(date_array, 0)
    mat_dict = dict.fromkeys(date_array, 0)

    for date in date_array:
        result_dict[date] = pd.DataFrame(columns=['date', 'c_code', tau, 'ground_truth'])
        result_dict[date]['c_code'] = stock_list
        result_dict[date]['date'] = date

    with torch.no_grad():
        for i in range(T):
            date = date_array[i]
            feature_tensor = np.load(f'{tensor_path}/{date}/feature.npy')
            X = torch.Tensor(feature_tensor).to(network.device)
            network_output = network.forward(X)
            result_dict[date][tau] = network_output.cpu().numpy()
            try:
                label_tensor = np.load(f'{tensor_path}/{date}/label.npy')
                result_dict[date]['ground_truth'] = label_tensor[:, 0]
            except FileNotFoundError:
                result_dict[date]['ground_truth'] = np.nan
            
            if graph:
                mat_dict[date] = network.encoder_graph(X).cpu().numpy()

    result_df = pd.concat(result_dict.values(), axis=0)
    return result_df, mat_dict, K


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=5e-4, 
                        help='Learning rate.')
    parser.add_argument('--lag', type=int, default=48, 
                        help='Number of lagged value of each feature')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units in encoder.')

    parser.add_argument('--N', type=int, default=2000, help='Number of assets in the simulation')
    parser.add_argument('--T', type=int, default=600, help='Number of time periods for the simulation')
    parser.add_argument('--K', type=int, default=3, help='Number of top nodes used for adjacency matrix construction')
    parser.add_argument('--dseed', type=int, default=0, help='Random seed for DGP')
    parser.add_argument('--cuda', type=int, default=0, help='Number of GPU device training on.')
    args = parser.parse_args()

    torch.cuda.set_device(args.cuda)
    
    model_path = f'{project_path}/simulated_quantile_model/{args.N}_{args.T}_{args.K}_{args.dseed}/hidden_{args.hidden}_lr_{args.lr}_lag_{args.lag}_horizon_1'
    tensor_path = f'{project_path}/simulated_tensor/{args.N}_{args.T}_{args.K}_{args.dseed}'
    moment_path = f'{project_path}/simulated_moment/{args.N}_{args.T}_{args.K}_{args.dseed}/hidden_{args.hidden}_lr_{args.lr}_lag_{args.lag}_horizon_1'
    tolerance = 20
    size = 0.01

    if not os.path.exists(moment_path):
        os.makedirs(moment_path)

    tau_list = os.listdir(model_path)
    tau_list.sort()
    tau_list = [float(x) for x in tau_list if isnumber(x)]

    date_list = os.listdir(tensor_path)
    date_array = np.array([int(x) for x in date_list])
    date_array.sort()
    stock_list = [f'stock_{x}' for x in range(2000)] # use integers to represent stock ids.
    
    inference_dict = dict.fromkeys(tau_list, 0)
    for tau in tqdm(tau_list, desc='inference'):
        result_df, mat_dict, K = infer(N=args.N,
                                       P=4,
                                       hidden_dim=args.hidden,
                                       tau=tau,
                                       date_array=date_array,
                                       stock_list=stock_list,
                                       model_path=model_path,
                                       tensor_path=tensor_path,
                                       graph=False)
        inference_dict[tau] = result_df

    for stock_name in tqdm(stock_list, desc='QCM regression'):
        try:
            df = compute_QCM_table(stock_name,
                                   inference_dict,
                                   tau_list,
                                   tolerance=tolerance,
                                   size=size,
                                   start_time=0,
                                   valid_time=399)
            df.to_csv(f'{moment_path}/{stock_name}.csv')
        except Exception as e:
            print(f'{stock_name} {e}')
