import numpy as np
import data as data
from matrix_operations import vec_matrix, mat_vec, replace_diagonal
from scipy.linalg import sqrtm


def initialize(number_regimes, no_lags, beta_hat=None):
    """
    :param number_regimes: Number of regimes in the model
    :param no_lags:  Number of lags in the VECM estimates
    :param beta_hat: Default None, uses journalese procedure to tabulate
    :return:
    delta_y_t: endogenous variable
    z_t_1: right-hand variable
    ols_resid:  residuals from the initial OLS estimates of the VECM model
    """

    delta_y_t, z_t_1, ols_resid = data.data_matrix(data.df, no_lags, beta_hat)
    k, obs = delta_y_t.shape

    # temp array to save u u.T values
    u_u = np.zeros([k * k, obs])
    # tabulate log squared residuals using the residuals
    for t in range(obs):
        u = ols_resid[:, t]
        u_u[:, t] = np.repeat(u, k) * np.tile(u, k)
    b_matrix = sqrtm(vec_matrix(u_u.sum(axis=1) / obs))
    b_matrix = b_matrix + np.random.normal(0, 1, size=(k, k))
    lam = replace_diagonal(np.random.normal(1, 0, size=k))
    sigma_array = np.zeros([number_regimes, k, k])
    for regime in range(number_regimes):
        if regime == 0:
            sigma_array[regime, :, :] = b_matrix @ b_matrix.T
        else:
            sigma_array[regime, :, :] = b_matrix @ lam @ b_matrix.T

    params = {'regimes': number_regimes,
              'epsilon_0': (np.log(np.ones(number_regimes) / number_regimes)).reshape(-1, 1),
              'transition_prob_mat': np.log(np.ones([number_regimes, number_regimes]) / number_regimes),
              'B_matrix': b_matrix,
              'lambda_m': np.identity(b_matrix.shape[0]),
              'sigma': sigma_array,
              'residuals': ols_resid,
              'VECM_params': None}
    return params, delta_y_t, z_t_1
