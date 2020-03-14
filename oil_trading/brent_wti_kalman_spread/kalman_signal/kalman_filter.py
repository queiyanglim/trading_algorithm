import pandas as pd
import numpy as np
from pykalman import KalmanFilter


def kalman_filter_regression(x, y, EM_on=False, EM_n_iter=5):
    """" Kalman Filter with choice to run Expectation-Maximization """
    # Transition Covariance
    delta = 1e-5
    trans_cov = delta * np.eye(2)

    # Observation matrix
    obs_mat = np.vstack([x, np.ones(x.shape)]).T[:, np.newaxis]

    kf = KalmanFilter(n_dim_obs=1,  # one observed value
                      n_dim_state=2,  # two states: slope and intercept
                      initial_state_mean=[0, 0],  # initiate means
                      initial_state_covariance=np.ones((2, 2)),  # initiate state covariances
                      transition_matrices=np.eye(2),  # identitiy matrix
                      observation_matrices=obs_mat,
                      observation_covariance=2,  # variance of y
                      transition_covariance=trans_cov)  # variance of coefficients

    if EM_on is True: kf = kf.em(y.values, n_iter=EM_n_iter)
    state_means, state_covs = kf.filter(y.values)
    return state_means


def kalman_filter_average(x):
    """ Kalman noise filtering on single series to extract hidden state."""

    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=.01)

    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    return state_means
