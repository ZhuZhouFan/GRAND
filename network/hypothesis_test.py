import numpy as np

def Kupic_test(prediction, ground_truth, tau):
    p = 1 - tau
    T = prediction.shape[0]
    N = np.sum(ground_truth > prediction)
    p_hat = N/T
    if p_hat == 0:
        p_hat += 1e-6
    if p_hat == 1:
        p_hat -= 1e-6
    log_likelihood_H0 = (T - N) * np.log(1 - p) + N * np.log(p)
    log_likelihood = (T - N) * np.log(1 - p_hat) + N * np.log(p_hat)
    test_statistic = -2 * (log_likelihood_H0 - log_likelihood)
    return test_statistic

def Christofer_test(prediction, ground_truth, tau):
    T = prediction.shape[0]
    I = 1.0 * (ground_truth > prediction)
    T_01 = np.sum((I[1:] - I[: -1]) == 1.0)
    T_10 = np.sum((I[1:] - I[: -1]) == -1.0)
    T_00 = np.sum(((I[1:] - I[: -1]) == 0.0) & (I[: -1] == 0.0))
    T_11 = np.sum(((I[1:] - I[: -1]) == 0.0) & (I[: -1] == 1.0))
    
    pi_hat = (T_01 + T_11)/T
    if T_00 + T_01 != 0.0:
        pi_hat_0 = T_01/(T_00 + T_01)
    else:
        pi_hat_0 = 0
    if T_10 + T_11 != 0.0:
        pi_hat_1 = T_11/(T_10 + T_11)
    else:
        pi_hat_1 = 0
    
    likelihood_H0 = (1 - pi_hat) ** (T_00 + T_10) * pi_hat ** (T_01 + T_11)
    likelihood = (1 - pi_hat_0) ** T_00 * pi_hat_0 ** T_01 * (1 - pi_hat_1) ** T_10 * pi_hat_1 ** T_11
    test_statistic_ind = -2 * np.log(likelihood_H0/likelihood)
    
    Kupic_stat = Kupic_test(prediction, ground_truth, tau)
    
    return test_statistic_ind + Kupic_stat