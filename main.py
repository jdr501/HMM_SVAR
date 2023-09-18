import numpy as np
import initialization as init
import expectation as exp
import optimization as op

# Model
regimes = 2
lags = 3
beta = np.array([0, 0, 0, 1]).reshape((-1, 1))  # must be shaped according to (number of variables, 1)
np.random.seed(42)
max_itr =30

# initialization
params, delta_y_t, z_t_1 = init.initialize(regimes, lags, beta_hat=beta)

x0 = [5.96082486e+01, 5.74334765e-01, 2.83277325e-01, 3.66479528e+00,
      -2.08881529e-01, 6.32170541e-04, -1.09137417e-01, -3.80763529e-01,
      4.24379418e+00, 1.83658083e-01, 2.16692718e-03, 1.29590368e+00,
      2.20826553e+00, -2.98484217e-01, -5.38269363e-03, 1.19668239e-03,
      0.012, 0.102, 0.843, 16.52]

for n in range(max_itr):
    print(f'this is n;{n}')
    smoothed_prob, state_joint = exp.expectation(params)

    params['epsilon_0'] = smoothed_prob[:, [0]]
    params['transition_prob_mat'], \
        params['B_matrix'], \
        params['sigma'], \
        params['VECM_params'], \
        params['residuals'], \
        x0 = op.m_step(state_joint, np.exp(smoothed_prob), x0, z_t_1, delta_y_t, params)

    print('=====trans prob mat ========')
    print(np.exp(params['transition_prob_mat']))
    print('=====VECM param2========')
