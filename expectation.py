import numpy as np
from scipy import stats


def logsumexp_(sum_series):
    max = np.max(sum_series)
    if max > 1e60 or max < -1e60:
        return max
    else:
        a = np.exp(sum_series[0] - max)
        for i in range(1, len(sum_series)):
            a += np.exp(sum_series[i] - max)
        return np.log(a) + max


def cond_prob_(param):
    obs = param['residuals'].shape[1]
    conditional_prob = np.zeros([param['regimes'], obs])  # y_t|s_t = j conditional density of Y for a given state
    for r in range(param['regimes']):
        conditional_prob[r, :] = stats.multivariate_normal(mean=None,
                                                           cov=param['sigma'][r, :, :]).logpdf(param['residuals'].T).T
    #print(np.exp(conditional_prob))
    return conditional_prob


def forward_(trans_prob, ln_eta_t, ini_dist):
    alpha = np.zeros((trans_prob.shape[0], ln_eta_t.shape[1]))
    alpha[:, [0]] = ini_dist * ln_eta_t[:, [0]]
    for t in range(1, ln_eta_t.shape[1]):
        for j in range(trans_prob.shape[0]):
            alpha_trans = alpha[:, t - 1] + trans_prob[:, j]
            alpha[j, t] = logsumexp_(alpha_trans) + ln_eta_t[j, t]
    #print(f'forward{alpha}')
    return alpha


def backward_(trans_prob, ln_eta_t):
    back = np.zeros((trans_prob.shape[0], ln_eta_t.shape[1]))
    # setting beta(T) = 1
    back[:, [ln_eta_t.shape[1] - 1]] = np.ones([trans_prob.shape[0], 1])
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(ln_eta_t.shape[1] - 2, -1, -1):
        for j in range(trans_prob.shape[0]):
            back_trans = ln_eta_t[j, [t + 1]] + trans_prob[j, :]
            back[j, t] = back[j, t + 1] + logsumexp_(back_trans)
    #print(f'back:{back}')
    return back


def smoothed_joint_(trans_prob, eta_t, alpha_hat, beta_hat):
    obs = alpha_hat.shape[1]
    regimes = alpha_hat.shape[0]
    alpha_beta = np.zeros(alpha_hat.shape)
    smoothed_prob = np.zeros(alpha_hat.shape)
    state_joint = np.zeros([regimes, regimes, obs - 1])

    for t in range(alpha_hat.shape[1]):
        for j in range(alpha_hat.shape[0]):
            alpha_beta[j, t] = alpha_hat[j, t] + beta_hat[j, t]
        smoothed_sum = logsumexp_(alpha_beta[:, t])
        smoothed_prob[:, [t]] = alpha_beta[:, [t]] - smoothed_sum

    for t in range(obs - 1):  # j at t k at t+1
        for k in range(regimes):
            for j in range(regimes):
                state_joint[j, k, t] = alpha_hat[j, t] + beta_hat[k, t + 1] + trans_prob[j, k] + eta_t[j, t + 1]
            sum_st_j = logsumexp_(state_joint[:, k, t])
            state_joint[:, k, t] = state_joint[:, k, t] - sum_st_j

    return smoothed_prob, state_joint


def expectation(param):
    """
    :param param: Dictionary of parameters
    :param ini_dist: Initial prob. to estimate alpha
    :return:
    smoothed prob
    joint probability of state j(t) and k(t+1).
    """
    ln_eta_t = cond_prob_(param)
    print(f'this is eta t: {ln_eta_t}')
    alpha_hat = forward_(param['transition_prob_mat'], ln_eta_t, param['epsilon_0'])
    beta_hat = backward_(param['transition_prob_mat'], ln_eta_t)
    smoothed_prob, state_joint = smoothed_joint_(param['transition_prob_mat'], ln_eta_t, alpha_hat, beta_hat, )
    #print(f'this is state joint prob:{state_joint}')
    return smoothed_prob, state_joint
