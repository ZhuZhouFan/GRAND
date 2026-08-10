"""
Estimate a Diebold-Yilmaz-style spillover index via a panel QVAR model.

Inputs:
  - Pipeline: ``valid_stocks.npy``, daily klines under ``kline_day/``.
  - External / related: ``{project_path}/kline_day_index/000001.XSHG.csv``
    (daily market calendar). Place this file under the data root if absent.
Operations: aggregate daily closes to monthly returns, fit a quantile VAR with
factors on a rolling in-sample window, and compute the FEVD spillover index
at each OOS month.
Outputs: ``{project_path}/macro_data/spillover_index_{tau}.csv`` consumed by
``bussiness_cycle.py``.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('.')
from config import project_path, start_time, valid_time, end_time

class QVAR(nn.Module):
    def __init__(self, N, T, p, f, tau=0.5, device=None):
        super().__init__()
        self.N = N
        self.T = T
        self.p = p
        self.f = f
        self.tau = tau
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.mu = nn.Parameter(torch.zeros(N, 1, device=self.device))
        self.Phi = nn.ParameterList([
            nn.Parameter(torch.zeros(N, N, device=self.device)) for _ in range(p)
        ])
        self.Lambda = nn.Parameter(torch.zeros(N, f, device=self.device))
        self.F = nn.Parameter(torch.zeros(T, f, 1, device=self.device))

    def forward(self, y_hist, t_idx):
        batch = y_hist.shape[0]
        y_pred = self.mu.repeat(batch, 1, 1)  # (batch, N, 1)
        for j in range(self.p):
            y_pred += torch.einsum('ij,bj->bi', self.Phi[j], y_hist[:, :, j]).unsqueeze(-1)
        f_t = self.F[t_idx]  # (batch, f, 1)
        y_pred += torch.einsum('nf,bfk->bnk', self.Lambda, f_t)
        return y_pred

    def quantile_loss(self, y_pred, y_true):
        diff = y_true - y_pred
        loss = torch.where(diff >= 0, self.tau * diff, (self.tau - 1) * diff)
        return loss.mean()

    def qr_identification(self):
        with torch.no_grad():
            F_matrix = self.F.squeeze(-1)  # (T, f)
            Q_F, R_F = torch.linalg.qr(F_matrix)  # Q_F: (T, f), R_F: (f, f)
            RF_Lambda = torch.matmul(R_F, self.Lambda.T)  # (f, N)
            Q_Lambda, R_Lambda = torch.linalg.qr(RF_Lambda)  # Q_Lambda: (N, f), R_Lambda: (f, f)
            self.F.data = (np.sqrt(self.T) * torch.matmul(Q_F, Q_Lambda.T)).unsqueeze(-1)
            self.Lambda.data = R_Lambda.T  # (N, f)

    def estimate_phi_lambda(self, y_hist, t_idx, y_target, lr=0.01, max_iter=100, tol=1e-6, patience=2):

        self.F.requires_grad = False
        for param in self.Phi:
            param.requires_grad = True
        self.Lambda.requires_grad = True
        self.mu.requires_grad = True
        

        params = [self.mu, self.Lambda] + list(self.Phi)
        optimizer = optim.Adam(params, lr=lr)
        
        prev_loss = float('inf')
        patience_counter = 0
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            y_pred = self.forward(y_hist, t_idx)
            loss = self.quantile_loss(y_pred, y_target)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            
            if iteration > 0:
                loss_decrease = prev_loss - current_loss
                if loss_decrease < tol:
                    patience_counter += 1
                else:
                    patience_counter = 0
                
                if patience_counter >= patience:
                    break

            prev_loss = current_loss
        
        self.F.requires_grad = True

    def estimate_f(self, y_hist, t_idx, y_target, lr=0.01, max_iter=100, tol=1e-6, patience=2):

        for param in self.Phi:
            param.requires_grad = False
        self.Lambda.requires_grad = False
        self.mu.requires_grad = False
        self.F.requires_grad = True
        
        optimizer = optim.Adam([self.F], lr=lr)
        
        prev_loss = float('inf')
        patience_counter = 0
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            y_pred = self.forward(y_hist, t_idx)
            loss = self.quantile_loss(y_pred, y_target)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            
            if iteration > 0:
                loss_decrease = prev_loss - current_loss
                if loss_decrease < tol:
                    patience_counter += 1
                else:
                    patience_counter = 0
                
                if patience_counter >= patience:
                    break

            prev_loss = current_loss
        
        for param in self.Phi:
            param.requires_grad = True
        self.Lambda.requires_grad = True
        self.mu.requires_grad = True

    def save_best_params(self):
        import copy
        self.best_params = {
            'mu': copy.deepcopy(self.mu.data),
            'Phi': [copy.deepcopy(phi.data) for phi in self.Phi],
            'Lambda': copy.deepcopy(self.Lambda.data),
            'F': copy.deepcopy(self.F.data)
        }
    
    def load_best_params(self):
        if hasattr(self, 'best_params'):
            with torch.no_grad():
                self.mu.data = self.best_params['mu'].clone()
                for i, phi in enumerate(self.Phi):
                    phi.data = self.best_params['Phi'][i].clone()
                self.Lambda.data = self.best_params['Lambda'].clone()
                self.F.data = self.best_params['F'].clone()
        else:
            raise ValueError("Best parameters not found")

    def fit(self, y_hist=None, t_idx=None, y_target=None, max_epochs=10, lr=0.001, tol=1e-3, patience=2):
        best_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(max_epochs):

            self.estimate_phi_lambda(y_hist, t_idx, y_target, lr=lr, tol=tol, patience=patience)

            self.estimate_f(y_hist, t_idx, y_target, lr=lr, tol=tol, patience=patience)

            self.qr_identification()
            
            with torch.no_grad():
                y_pred = self.forward(y_hist, t_idx)
                total_loss = self.quantile_loss(y_pred, y_target)
                current_loss = total_loss.item()

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_epoch = epoch + 1
                    self.save_best_params()

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    break

        self.load_best_params()

        return best_loss, best_epoch

    def compute_residuals(self, y_hist, t_idx, y_target):
        """Return residuals ``v_t = y_t - y_pred`` with shape ``(T, N)``."""
        with torch.no_grad():
            y_pred = self.forward(y_hist, t_idx)
            residuals = y_target - y_pred
            return residuals.squeeze(-1)

    def compute_omega_matrix(self, y_hist, t_idx, y_target):
        residuals = self.compute_residuals(y_hist, t_idx, y_target)

        residuals_centered = residuals - residuals.mean(dim=0, keepdim=True)
        omega = torch.matmul(residuals_centered.T, residuals_centered) / (self.T - 1)

        return omega

    def compute_wold_coefficients(self, max_horizon=20):
        B_coeffs = []

        B_0 = torch.eye(self.N, device=self.device)
        B_coeffs.append(B_0)

        for j in range(1, max_horizon + 1):
            B_j = torch.zeros(self.N, self.N, device=self.device)

            for k in range(1, min(j + 1, self.p + 1)):
                if j - k >= 0:
                    B_j += torch.matmul(self.Phi[k - 1], B_coeffs[j - k])

            B_coeffs.append(B_j)

        return B_coeffs

    def compute_fevd_matrix(self, y_hist, t_idx, y_target, horizon, max_wold_horizon=None):
        if max_wold_horizon is None:
            max_wold_horizon = horizon

        with torch.no_grad():

            omega = self.compute_omega_matrix(y_hist, t_idx, y_target)
            B_coeffs = self.compute_wold_coefficients(max_wold_horizon)
            fevd_matrix = torch.zeros(self.N, self.N, device=self.device)
            for j in range(self.N):

                e_j = torch.zeros(self.N, device=self.device)
                e_j[j] = 1.0

                denominator = 0.0
                for l in range(horizon + 1):
                    if l < len(B_coeffs):
                        B_l = B_coeffs[l]
                        term = torch.matmul(e_j, torch.matmul(B_l, torch.matmul(omega, torch.matmul(B_l.T, e_j))))
                        denominator += term

                for i in range(self.N):
                    e_i = torch.zeros(self.N, device=self.device)
                    e_i[i] = 1.0

                    omega_ii = omega[i, i]
                    numerator = 0.0

                    for l in range(horizon + 1):
                        if l < len(B_coeffs):
                            B_l = B_coeffs[l]

                            numerator += torch.matmul(e_j, torch.matmul(B_l, torch.matmul(omega, e_i))) ** 2

                    numerator = numerator / omega_ii

                    fevd_matrix[i, j] = numerator / denominator

        return fevd_matrix

    def compute_spillover_index(self, y_hist, t_idx, y_target, horizon, max_wold_horizon=None):
        fevd_matrix = self.compute_fevd_matrix(y_hist, t_idx, y_target, horizon, max_wold_horizon)
        
        row_sums = fevd_matrix.sum(dim=0)
        spillover_matrix = torch.zeros_like(fevd_matrix)
        for j in range(self.N):
            if row_sums[j] > 1e-10:
                spillover_matrix[:, j] = fevd_matrix[:, j] / row_sums[j]
        
        from_degree = torch.zeros(self.N, device=self.device)
        for i in range(self.N):
            from_degree[i] = spillover_matrix[i, :].sum() - spillover_matrix[i, i]

        spillover_index = from_degree.mean()
        
        return spillover_index.item()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=3, help='QVAR lag order')
    parser.add_argument('--f', type=int, default=5, help='Number of latent factors')
    parser.add_argument('--tau', type=float, default=0.05, help='Quantile level')
    parser.add_argument('--cuda', type=int, default=0, help='GPU device index')
    args = parser.parse_args()

    index_day_path = f'{project_path}/kline_day_index/000001.XSHG.csv'
    if not os.path.exists(index_day_path):
        raise FileNotFoundError(
            f'Daily index calendar not found: {index_day_path}. '
            f'Place 000001.XSHG.csv under {{project_path}}/kline_day_index/.')

    index_day = pd.read_csv(index_day_path, index_col='date')
    selected_dates = index_day.loc[start_time:end_time, :].index.values
    stock_list = np.load(f'{project_path}/valid_stocks.npy', allow_pickle=True)

    close_matrix = pd.DataFrame(index=selected_dates, columns=stock_list)
    for stock in stock_list:
        try:
            kline_day = pd.read_csv(f'{project_path}/kline_day/{stock}.csv', index_col='date')
        except FileNotFoundError:
            continue
        close_matrix[stock] = kline_day['close']

    close_matrix.index = pd.to_datetime(close_matrix.index, format='%Y-%m-%d')
    monthly_prices = close_matrix.resample('ME').last()
    monthly_returns = monthly_prices.pct_change(fill_method=None)
    monthly_returns.dropna(how='all', inplace=True)
    monthly_returns = monthly_returns.apply(lambda row: row.fillna(row.median()), axis=1)

    N = len(stock_list)
    p = args.p
    f = args.f
    tau = args.tau
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'

    oos_dates = monthly_returns.loc[valid_time:end_time, :].index.values
    spillover_index = pd.DataFrame(index=oos_dates, columns=['index'])
    spillover_index.index.name = 'date'

    os.makedirs(f'{project_path}/macro_data', exist_ok=True)
    out_path = f'{project_path}/macro_data/spillover_index_{tau}.csv'

    for date in tqdm(oos_dates):

        previous_date = date - pd.Timedelta(weeks=52 * 10)
        insample_returns = monthly_returns.loc[previous_date:date, :].values
        T = insample_returns.shape[0] - p

        y_hist = torch.zeros(T, N, p)
        y_target = torch.zeros(T, N, 1)
        t_idx = torch.arange(T)

        for t in range(T):
            y_target[t, :, 0] = torch.tensor(insample_returns[t + p, :])
            for j in range(p):
                y_hist[t, :, j] = torch.tensor(insample_returns[t + p - 1 - j, :])

        y_hist = y_hist.to(device)
        t_idx = t_idx.to(device)
        y_target = y_target.to(device)

        qvar = QVAR(N, T, p, f, tau, device)
        best_loss, best_epoch = qvar.fit(y_hist, t_idx, y_target,
                                         max_epochs=20, lr=1e-5)

        spillover_index.loc[date, 'index'] = qvar.compute_spillover_index(
            y_hist, t_idx, y_target, horizon=1)
        spillover_index.to_csv(out_path)

    print(f'Saved: {out_path}')