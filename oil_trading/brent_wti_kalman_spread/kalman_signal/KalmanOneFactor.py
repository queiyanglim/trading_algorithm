from pykalman import KalmanFilter
import numpy as np
import pandas as pd
from pnl_process.periodic_settlement import periodic_settlement
from oil_trading.data.prepare_reuters_data import get_reuters_data


class KalmanOneFactor:
    def __init__(self, initial_capital, price_series, x_name, y_name, lower_bound, upper_bound, train_period):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.pnl_vector = None
        self.initial_capital = initial_capital
        self.x, self.y = price_series[x_name], price_series[y_name]
        self.train_period = train_period
        # Main DataFrame to work on
        self.df = pd.DataFrame({"y": self.y, "x": self.x})
        self.df.index = pd.to_datetime(price_series.index)
        self.kf = None
        self.trained = False

    def backtest(self):
        if self.trained is False:
            self.train_dataset(self.train_period)
        self.trading_rule()
        self.dynamic_unit_allocation()
        self.settlement()
        return self.df

    def train_dataset(self, train_period, print_covariance=True):
        self.kf, res = kalman_filter_one_factor_trained(self.x, self.y, train_period)
        self.df["hr"] = -res.state_mean
        self.df["hedged_spread"] = self.y + self.x * self.df.hr
        self.trained = True
        self.df = self.df.iloc[train_period+1:]
        if print_covariance is True:
            print("observation_covariance:", self.kf.observation_covariance)
            print("transition_covariance:", self.kf.transition_covariance)

    def trading_rule(self):
        if self.trained:
            hedged_spread = self.df.hedged_spread
            win = 10
            z_score = (hedged_spread - hedged_spread.rolling(window=win).mean())/hedged_spread.rolling(window=win).std()

            long_spread = (z_score < self.lower_bound) * 1
            short_spread = (z_score > self.upper_bound) * -1
            self.df["long_short_spread"] = long_spread + short_spread
            self.df["z_score"] = z_score
            self.df = self.df.dropna()

    def dynamic_unit_allocation(self):
        """ Dynamically adjust position """
        self.df["y_pos"] = np.floor(self.df.long_short_spread * self.initial_capital / self.df.y)
        self.df["x_pos"] = np.floor(self.df.hr * self.df.y_pos)

    def settlement(self):
        pnl_y = periodic_settlement(self.df, "y", "y_pos")
        pnl_x = periodic_settlement(self.df, "x", "x_pos")
        self.pnl_vector = pnl_x + pnl_y


# one factor kalman
def kalman_filter_one_factor_trained(x_series, y_series, train_period):
    x_train = x_series[:train_period]
    y_train = y_series[:train_period]
    x_train = np.vstack([x_train]).T[:, np.newaxis]  # shape = (N,1,1)
    y_train = np.vstack([y_train])  # shape = (1, N)

    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=x_train,
                      initial_state_mean=[1])
    kf.em(y_train)
    state_mean, state_cov = kf.filter(y_train)
    x_predict = x_series[train_period:]
    y_predict = y_series[train_period:]

    predict = {}
    i = 0
    state_mean_cache, state_cov_cache = state_mean[-1], state_cov[-1]
    for ind, x, y in zip(x_predict.index, x_predict, y_predict):
        res = kf.filter_update([state_mean_cache], state_cov_cache, observation=y, observation_matrix=np.array([[x]]))
        state_mean_cache = res[0][0]
        state_cov_cache = res[1]
        predict[i] = {x_series.index.name: ind,
                      "state_mean": float(state_mean_cache),
                      "state_cov": float(state_cov_cache)}
        i = i + 1
    return kf, pd.DataFrame(predict).T.set_index(x_series.index.name)
