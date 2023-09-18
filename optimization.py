from scipy.linalg import pinv
from scipy.optimize import minimize

from expectation import logsumexp_
from scipy import stats  #
import numpy as np
from matrix_operations import vec_matrix, replace_diagonal
from sklearn.covariance import LedoitWolf


def trans_prob_mat(state_joint):
    regimes = state_joint.shape[0]
    obs = state_joint.shape[2]
    trans_prob = np.zeros([regimes, regimes])
    gamma = np.zeros([regimes, obs])
    gamma_sum = np.zeros([regimes, 1])

    for t in range(obs):
        for j in range(regimes):
            gamma[j, t] = logsumexp_(state_joint[:, j, t])
    for j in range(regimes):
        gamma_sum[j, 0] = logsumexp_(gamma[j, :])

    for i in range(regimes):
        for j in range(regimes):
            trans_prob[i, j] = logsumexp_(state_joint[i, j, :]) - gamma_sum[j, 0]

    return trans_prob


def marginal_density(x, opt_parameters):
    regimes, k, residuals, smoothed_prob = opt_parameters
    b_mat = vec_matrix(np.array(x[0:k ** 2]))
    lam_m = np.zeros([regimes - 1, k, k])
    mean_zero = np.zeros([k])

    start = k * k
    for m in range(regimes - 1):
        end = start + k
        lam_m[m, :, :] = replace_diagonal(x[start:end])
        start = end

    # create an array of lambdas here ...
    # also include restricted models  ....
    likelihood_array = np.zeros([regimes])
    sigma = np.zeros([regimes, k, k])
    for regime in range(regimes):
        if regime == 0:
            sigma[regime, :, :] = b_mat @ b_mat.T
        else:
            sigma[regime, :, :] = b_mat @ lam_m[regime - 1, :, :] @ b_mat.T

        x = np.random.multivariate_normal(mean=mean_zero, cov=sigma[regime, :, :], size=50) # , check_valid='ignore'
        cov = LedoitWolf().fit(x)
        sigma[regime, :, :] = cov.covariance_

    for regime in range(regimes):
        # Note that I am taking the exp of smoothed probability because original smoothed prob. are in log this
        likelihood_array[regime] = (np.exp(smoothed_prob[regime, :]) * stats.multivariate_normal(
            mean=None, cov=sigma[regime, :, :], allow_singular=True).logpdf(residuals.T).T).sum()
    return -likelihood_array.sum()


def b_matrix_sigma(x, k, regimes):
    b_mat = vec_matrix(np.array(x[0:k ** 2]))
    lam_m = np.zeros([regimes - 1, k, k])
    start = k * k
    for m in range(regimes - 1):
        end = start + k
        lam_m[m, :, :] = replace_diagonal(x[start:end])
        start = end

    sigma = np.zeros([regimes, k, k])
    for regime in range(regimes):
        if regime == 0:
            sigma[regime, :, :] = b_mat @ b_mat.T
        else:
            sigma[regime, :, :] = b_mat @ lam_m[regime - 1, :, :] @ b_mat.T

    return b_mat, sigma


def m_step(state_joint, smoothed_prob, x0, zt, delta_y, parameters):
    # optimization additional arguments (tuple)
    print('========Optimization=========')
    k, obs = parameters['residuals'].shape

    ####################################
    # estimating transition probability
    ####################################

    transition_prob_mat = trans_prob_mat(state_joint)

    print(f'this is transition prob mat:{transition_prob_mat} ')

    ####################################
    # estimating Covariance matrices
    ####################################

    # bounds to ensure positive semi-definite
    bound_list = []
    for i in range(len(x0)):
        if i < k ** 2:
            bound_list.append((None, None))
        else:
            bound_list.append((0.01, None))
    bound_list = tuple(bound_list)

    # no need to take  exponential of smoothed prob as optimization file does it there.
    op_params = [parameters['regimes'], k, parameters['residuals'], smoothed_prob]

    res = minimize(marginal_density, x0, args=op_params,
                   bounds=bound_list,  method='Nelder-Mead', options={'maxiter': 15000, 'disp': False})


    print(res.message)
    b_mat, sigma = b_matrix_sigma(res.x, k, parameters['regimes'])


    ####################################
    # estimate weighted least-square parameters
    ####################################

    for regime in range(parameters['regimes']):
        t_sum = np.zeros([zt.shape[0], zt.shape[0]])
        m_sum = np.zeros([zt.shape[0] * k, zt.shape[0] * k])
        m_sum_numo = np.zeros([zt.shape[0] * k, k])
        t_sum_numo = np.zeros([zt.shape[0] * k, 1])

        for t in range(zt.shape[1]):
            t_sum += np.exp(smoothed_prob[regime, t]) * zt[:, [t]] @ zt[:, [t]].T
        m_sum += np.kron(t_sum, pinv(sigma[regime, :, :]))
        denominator = pinv(m_sum)

    for t in range(zt.shape[1]):
        for regime in range(parameters['regimes']):
            m_sum_numo += np.kron(np.exp(smoothed_prob[regime, t]) * zt[:, [t]], pinv(sigma[regime, :, :]))
        t_sum_numo += m_sum_numo @ delta_y[:, [t]]

    theta_hat = denominator @ t_sum_numo

    ####################################
    # residuals estimate
    ####################################

    resid = np.zeros(delta_y.shape)
    for t in range(zt.shape[1]):
        resid[:, [t]] = delta_y[:, [t]] - np.kron(zt[:, [t]].T, np.identity(delta_y.shape[0])) @ theta_hat

    return transition_prob_mat, b_mat, sigma, theta_hat, resid, res.x
