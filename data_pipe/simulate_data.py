import numpy as np
import torch
import argparse
from tqdm import tqdm
import os
from joblib import Parallel, delayed

import sys
sys.path.append('.')
from config import project_path
from network.NCM import nearcorr, cov2cor, cor2cov

class GRANDSimulation:
    def __init__(self,
                 N=2000,
                 T=600,
                 alpha=0.9,
                 beta=0.6,
                 zeta=0.2,
                 gamma=0.6,
                 delta=0.9,
                 K_star=5,
                 seed=42,
                 burnin=300):

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.N = N
        self.T = T
        self.burnin = burnin

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.zeta = zeta

        self.K_star = K_star

        self.rho_k = np.random.uniform(0.8, 1, size=3)
        self.Z = np.zeros((N, T + burnin, 3))
        self.r = np.zeros((N, T + burnin))
        self.epsilon = np.random.normal(0, 1, (N, T + burnin))

        self.mask_row = torch.linspace(0, N - 1, N).reshape([-1, 1]).repeat(1, K_star).reshape(1, -1).long()
        self.off_diagonal_mat = (torch.ones(N, N) - torch.eye(N))

    def alpha_function(self, z, r_prev):
        return 0.1 * r_prev * z[:, 0] + 0.1 * z[:, 0] ** 2 + 0.5 * np.cos(z[:, 1]) + 0.5 * np.sign(z[:, 2])
        # return 0

    def simulate_characteristics(self):
        for k in range(3):
            for t in range(1, self.T + self.burnin):
                self.Z[:, t, k] = self.rho_k[k] * self.Z[:, t-1, k] + np.random.normal(0, 1, self.N)

    def cal_cos_similarity(self, x, y):
        '''
        brorrowed from the model.py
        '''
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x*x, dim=1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y*y, dim=1)).reshape(-1, 1)
        cos_similarity = xy/x_norm.mm(torch.t(y_norm))
        cos_similarity[cos_similarity != cos_similarity] = 0
        return cos_similarity

    def construct_W_t(self, z):
        '''
        brorrowed from the model.py
        '''
        z_ = torch.Tensor(z)
        similarity_mat = self.cal_cos_similarity(z_, z_)
        similarity_mat = similarity_mat * self.off_diagonal_mat

        mask_column = torch.topk(similarity_mat, self.K_star, dim=1)[1].reshape(1, -1)
        mask = torch.zeros([self.N, self.N])
        mask[self.mask_row, mask_column] = 1
        topK_similarity_mat = similarity_mat * mask
        topK_similarity_mat = topK_similarity_mat[:, topK_similarity_mat.sum(0) != 0]
        
        non_zero_counts = (topK_similarity_mat != 0).sum(dim=0)
        selected_columns = non_zero_counts > 1
        topK_similarity_mat_ = topK_similarity_mat[:, selected_columns]
        
        # adj_mats = torch.zeros([self.N, self.N, topK_similarity_mat_.shape[1]])
        adj_mat = torch.zeros([self.N, self.N])
        for concept_idx in range(topK_similarity_mat_.shape[1]):
            non_zero_idx = topK_similarity_mat_[:, concept_idx]
            connected_stocks = torch.where(non_zero_idx)[0]

            for j, sender in enumerate(connected_stocks[:-1]):
                receiver = connected_stocks[j+1]
                # adj_mats[sender, receiver, concept_idx] = 1
                # adj_mats[receiver, sender, concept_idx] = 1
                adj_mat[sender, receiver, concept_idx] = 1
                adj_mat[receiver, sender, concept_idx] = 1
        
        # adj_matrix_sum = adj_mats.sum(dim=-1)
        # row_sums = adj_matrix_sum.sum(dim=1, keepdim=True)
        # W_t = 0.1 * adj_matrix_sum / row_sums
        
        # W_t = adj_mats.mean(axis = -1)
        # W_t[W_t > 0] = 1
        
        W_t = adj_mat
        
        return W_t.numpy()

    def simulate(self):

        self.simulate_characteristics()

        Sigmas = np.zeros((self.N, self.N, self.T + self.burnin))
        graphs = np.zeros((self.N, self.N, self.T + self.burnin))

        # pre_Sigmas = np.zeros((self.N, self.N, self.T + self.burnin))
        # for t in range(self.T + self.burnin):
        # np.fill_diagonal(pre_Sigmas[:, :, t], 1.0)

        pre_Sigmas = np.full((self.N, self.N, self.T + self.burnin), 0.25)
        diag_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0]
        for t in range(self.T + self.burnin):
            random_diagonals = np.random.choice(diag_values, size=self.N)
            np.fill_diagonal(pre_Sigmas[:, :, t], random_diagonals)

        iden_mat = np.eye(self.N, self.N)

        for t in range(round(0.5 * self.burnin), self.T + self.burnin):

            W_t = self.construct_W_t(self.Z[:, t, :])
            graphs[:, :, t] = W_t
            sigma_diag = np.diag(pre_Sigmas[:, :, t-1])
            epsilon_square = np.outer(self.epsilon[:, t-1], self.epsilon[:, t-1])

            pre_Sigmas[:, :, t] = (
                0.05 * iden_mat
                + self.alpha * np.diag(sigma_diag)
                + self.beta * np.diag(W_t @ sigma_diag)
                + self.gamma * (epsilon_square * iden_mat)
                + W_t * (self.delta * epsilon_square +
                         self.zeta * pre_Sigmas[:, :, t-1])
            )

            if self.is_positive_definite(pre_Sigmas[:, :, t]):
                Sigmas[:, :, t] = pre_Sigmas[:, :, t]
            else:
                Sigmas[:, :, t] = self.nearest_PSD(pre_Sigmas[:, :, t])

            epsilon_t = np.random.multivariate_normal(np.zeros(self.N), Sigmas[:, :, t])
            self.epsilon[:, t] = epsilon_t

            z_prev = self.Z[:, t-1, :]
            r_prev = self.r[:, t-1]
            self.r[:, t] = self.alpha_function(z_prev, r_prev) + epsilon_t

        return self.r, self.Z, self.epsilon, Sigmas, graphs

    def nearest_PSD(self, pre_sigma):
        correlation_mat = cov2cor(pre_sigma)
        try:
            pd_correlation_mat = nearcorr(correlation_mat, max_iterations=1e3)
            pd_covariance_mat = cor2cov(
                pd_correlation_mat, np.diagonal(pre_sigma))
            return pd_covariance_mat
        except Exception as e:
            print(e)

    def is_positive_definite(self, matrix):
        eigenvalues = np.linalg.eigvalsh(matrix)
        return np.all(eigenvalues > 0)

def subjob(args, seed):
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    
    simulator = GRANDSimulation(N=args.N,
                                T=args.T,
                                alpha=args.alpha,
                                beta=args.beta,
                                gamma=args.gamma,
                                delta=args.delta,
                                zeta=args.zeta,
                                K_star=args.K_star,
                                seed=seed,
                                burnin=args.burnin)

    r, Z, epsilon, Sigmas, graphs = simulator.simulate()
    
    save_log = f'{project_path}/simulated_tensor/{args.N}_{args.T}_{args.K_star}_{seed}'

    for t in range(args.burnin, args.burnin + args.T):
        r_prev = np.expand_dims(r[:, t-args.lag:t], axis = -1)
        Z_prev = Z[:, t-args.lag:t, :]
        feature = np.concatenate([Z_prev, r_prev], axis = -1)
        label = np.expand_dims(r[:, t], axis = -1)
        Sigma = Sigmas[:, :, t]
        W = graphs[:, :, t]
        
        save_log_ = f'{save_log}/{t - args.burnin}'
        os.makedirs(save_log_, exist_ok=True)
        
        np.save(f'{save_log_}/feature.npy', feature)
        np.save(f'{save_log_}/label.npy', label)
        np.save(f'{save_log_}/Sigma.npy', Sigma)
        np.save(f'{save_log_}/W.npy', W)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GRAND Simulator Hyperparameters')
    parser.add_argument('--N', type=int, default=200, help='Number of assets in the simulation')
    parser.add_argument('--T', type=int, default=600, help='Number of time periods for the simulation')
    parser.add_argument('--alpha', type=float, default=0.8, help='Alpha parameter, controlling the asset return dynamics')
    parser.add_argument('--beta', type=float, default=0.025, help='Beta parameter, influencing the asset correlation structure')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma parameter, affecting the idiosyncratic variance')
    parser.add_argument('--delta', type=float, default=0.2, help='Delta parameter, influencing the cross-sectional covariance')
    parser.add_argument('--zeta', type=float, default=0.2, help='Zeta parameter, controlling the correlation decay rate')
    parser.add_argument('--K_star', type=int, default=3, help='Number of top nodes used for adjacency matrix construction')
    # parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--burnin', type=int, default=100, help='Burn-in period to discard initial simulation data')
    parser.add_argument('--lag', type=int, default=48, help='Number of lagged values to use for each feature')
    parser.add_argument('--rep', type=int, default=100, help='Number of Monte Carlo repetitions for the simulation')
    parser.add_argument('--cpu', type=int, default=20, help='Number of CPUs')
    args = parser.parse_args()
    
    Parallel(n_jobs=args.cpu)(delayed(subjob)(args, seed) for seed in tqdm([x for x in range(args.rep)]))
    
    # for seed in tqdm([x for x in range(100)]):
    #     subjob(args, seed)
    #     break