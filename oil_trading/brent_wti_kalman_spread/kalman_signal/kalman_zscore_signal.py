from pykalman import KalmanFilter
from statsmodels.distributions.empirical_distribution import ECDF
from pnl_process.periodic_settlement import periodic_settlement
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


class KalmanFilterZScoreSignal:
    def __init__(self, initial_capital, price_series, x_name, y_name, entry_z_score, exit_z_score,
                 z_score_rolling_window):
        """
        Take a df with x and y (indexed with timestamp) and
        return a df with ZScore signal.

        Long spread is defined as long y and short x
        i.e. modelled as y = ax + b
        *x_name to be called using df[x_name] for its series
        """
        self.pnl_vector = None
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.z_score_rolling_window = z_score_rolling_window
        self.initial_capital = initial_capital
        self.x, self.y = price_series[x_name], price_series[y_name]

        # Main DataFrame to work on
        self.df = pd.DataFrame({"y": self.y, "x": self.x})
        self.df.index = pd.to_datetime(price_series.index)

        self.state_means, self.state_covs = kalman_filter_regression(x=self.x, y=self.y)
        self.df["hr"] = -self.state_means[:, 0]
        self.df["intercept"] = self.state_means[:, 1]
        self.df["kalman_hedged_spread"] = self.df.y + self.df.hr * self.df.x
        self.mean_spread = None
        self.std_spread = None

    # Definte strategy pipeline here
    def backtest_strategy(self):
        self.z_score_signal_fixed_rolling_window()
        self._z_score_trading_logic()
        self.dynamic_unit_allocation()
        self.settlement()
        return self.df

    def settlement(self):
        pnl_y = periodic_settlement(self.df, "y", "y_pos")
        pnl_x = periodic_settlement(self.df, "x", "x_pos")
        self.pnl_vector = pnl_x + pnl_y

    ###################################################################################################################
    # Determine Z score signals to filter
    # version 1: Fixed Rolling Window
    def z_score_signal_fixed_rolling_window(self):
        self.mean_spread = self.df.kalman_hedged_spread.rolling(window=self.z_score_rolling_window).mean()
        self.std_spread = self.df.kalman_hedged_spread.rolling(window=self.z_score_rolling_window).std()

        self.mean_spread.name = "mean_spread"
        self.std_spread.name = "std_spread"
        self.df["z_score"] = (self.df.kalman_hedged_spread - self.mean_spread) / self.std_spread

    # version 2: Volatility controlled window
    # Dynamically select rolling mean and std window based on volatility of price series y
    # Sample window size based on empirical CDF of historical volatility
    # If volatility is high, then calculate z_score using with more weight in front end data
    # If volatility is low, then calculate z_score from smoother rolling mean/std
    # Caveat: Looked forward to build empirical CDF so not realistic per se
    def vol_control_window_selection(self):
        vol_rolling_window = self.z_score_rolling_window
        vol_data = self.y
        vol_data = vol_data.diff().ewm(span=vol_rolling_window).std()
        vol_ecdf = ECDF(vol_data.values)

        def window_sample(x, ecdf):
            return np.floor((1 - ecdf(x)) * 12 + 2)

        self.df["aug_window"] = window_sample(vol_data)

        # Create a z-score dataframe containing all columns of possible windows
        all_possible_windows = map(int, list(set(self.df.aug_window.values)))

        data = self.df.kalman_hedged_spread.copy()
        dynamic_rolling_mean = pd.DataFrame(index=data.index)
        dynamic_rolling_std = dynamic_rolling_mean.copy()

        for window in all_possible_windows:
            dynamic_rolling_mean[f"window_{window}"] = data.rolling(window=window).mean()
            dynamic_rolling_std[f"window_{window}"] = data.rolling(window=window).std()

        for s in dynamic_rolling_mean.iterrows():
            roll_window = "window_" + str(int(self.df.loc[s[0], "aug_window"]))
            self.df.loc[s[0], "dy_roll_mean"] = s[1][roll_window]

        for s in dynamic_rolling_std.iterrows():
            roll_window = "window_" + str(int(self.df.loc[s[0], "aug_window"]))
            self.df.loc[s[0], "dy_roll_std"] = s[1][roll_window]

        self.df["z_score"] = (self.df.hedged_spread.values - self.df.dy_roll_mean.values) / self.df.dy_roll_std.values
        self.mean_spread = self.df.dy_roll_mean
        self.std_spread = self.df.dy_roll_std

        # Show window sampling profile
        plot_space = np.arange(0, vol_data.max(), 0.01)
        plt.plot(plot_space, vol_ecdf(plot_space))
        plt.title("Window sampling profile")
        plt.show()

    # version 3: Use Chi2 or Kolmogorov-Smirnov to sample rolling window size from historical volatility distribution
    def fitted_distribution_window_size_z_score_signal(self):
        # dist = stats.chi2
        dist = stats.kstwobign

        # Annualized factor. If day then == 1, if hour then == 1/24 and so on.
        ann_scale = float(stats.mode((np.diff(self.df.index) / np.timedelta64(1, 'D')))[0])
        # vol_data = self.df.y.pct_change().rolling(window=10).std(ddof=1) * np.sqrt((1 / ann_scale) * 252)
        vol_data = self.df.y.pct_change().ewm(span=self.z_score_rolling_window).std(ddof=1) * np.sqrt(
            (1 / ann_scale) * 252)
        vol_data = vol_data.fillna(0)

        params = dist.fit(vol_data)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        def pdf_x(x):
            return np.floor(dist.pdf(x, loc=loc, scale=scale, *arg) * scale * 8) + 1

        # Plot window profile
        s = np.arange(0, vol_data.max(), 0.01)
        plt.plot(s, pdf_x(s))
        plt.title("Rolling Windoow Profile")
        plt.show()
        self.df["aug_window"] = pdf_x(vol_data)

        # Create a z-score dataframe containing all columns of possible windows
        all_possible_windows = map(int, list(set(self.df.aug_window.values)))

        data = self.df.hedged_spread.copy()
        dynamic_rolling_mean = pd.DataFrame(index=data.index)
        dynamic_rolling_std = dynamic_rolling_mean.copy()

        for window in all_possible_windows:
            dynamic_rolling_mean[f"window_{window}"] = data.rolling(window=window).mean()
            dynamic_rolling_std[f"window_{window}"] = data.rolling(window=window).std()

        # dynamic_rolling_mean.tail(20).plot(figsize=(20,15))
        for s in dynamic_rolling_mean.iterrows():
            roll_window = "window_" + str(int(self.df.loc[s[0], "aug_window"]))
            self.df.loc[s[0], "dy_roll_mean"] = s[1][roll_window]

        for s in dynamic_rolling_std.iterrows():
            roll_window = "window_" + str(int(self.df.loc[s[0], "aug_window"]))
            self.df.loc[s[0], "dy_roll_std"] = s[1][roll_window]

        self.df["z_score"] = (self.df.hedged_spread.values - self.df.dy_roll_mean.values) / self.df.dy_roll_std.values
        self.mean_spread = self.df.dy_roll_mean
        self.std_spread = self.df.dy_roll_std

    ###################################################################################################################
    # Process long or short spread position based on z score signal
    def _z_score_trading_logic(self):
        # Process position after calling self._generate_z_score_signal()
        if "z_score" not in list(self.df.columns):
            raise Exception("z_score not initialized")

        # Enter long position if current z-score is less than entry threshold AND previous day was within threshold
        # bound. Exit long position if previous day was less than exit score and current score is larger than exit
        # score
        self.df["long_entry"] = (
                (self.df.z_score < -self.entry_z_score) & (self.df.z_score.shift(1) > - self.entry_z_score))
        self.df["long_exit"] = (
                (self.df.z_score > -self.exit_z_score) & (self.df.z_score.shift(1) < - self.exit_z_score))

        # Enter short position if otherwise
        self.df["short_entry"] = (
                (self.df.z_score > self.entry_z_score) & (self.df.z_score.shift(1) < self.entry_z_score))
        self.df["short_exit"] = ((self.df.z_score < self.exit_z_score) & (self.df.z_score.shift(1) > self.exit_z_score))

        # Set up positions
        self.df["num_units_long"] = np.nan
        self.df["num_units_short"] = np.nan
        self.df.loc[self.df.long_entry, "num_units_long"] = 1
        self.df.loc[self.df.long_exit, "num_units_long"] = 0
        self.df.loc[self.df.short_entry, "num_units_short"] = -1
        self.df.loc[self.df.short_exit, "num_units_short"] = 0
        self.df.loc[self.df.index[0], ["num_units_long", "num_units_short"]] = 0

        # Forward filling and establish "long_short_spread"
        self.df["num_units_long"] = self.df["num_units_long"].fillna(method="pad")
        self.df["num_units_short"] = self.df["num_units_short"].fillna(method="pad")
        self.df["long_short_spread"] = self.df.num_units_short + self.df.num_units_long
        self.df.drop(["num_units_long", "num_units_short"], axis=1)

    ###################################################################################################################
    # Long or short units allocation. These are called y_pos and x_pos respectively
    # Positive sign represents long and negative sign represents short
    def dynamic_unit_allocation(self):
        """ Dynamically adjust position """
        self.df["y_pos"] = np.floor(self.df.long_short_spread * self.initial_capital / self.df.y)
        self.df["x_pos"] = np.floor(self.df.hr * self.df.y_pos)

    def static_unit_allocation(self):
        """
        Do not reposition positions based on dynamic
        """
        self.df["enter"] = (self.df.long_short_spread != self.df.long_short_spread.shift(1)) & (
                self.df.short_entry | self.df.long_entry)
        self.df = self.df[["y", "x", "hr", "long_short_spread", "enter"]]

        self.df["x_pos"] = np.nan
        self.df["y_pos"] = np.nan
        self.df = self.df.iloc[40:]  # FIXME: may not be necessary
        self.df.loc[(self.df.long_short_spread == 0), ["x_pos", "y_pos"]] = 0

        self.df.loc[self.df.enter, "y_pos"] = self.df.long_short_spread * self.initial_capital / self.df.y.loc[
            self.df.enter]
        self.df.loc[self.df.enter, "x_pos"] = self.df.y_pos * self.df.hr
        self.df.x_pos = self.df.x_pos.fillna(method="pad")
        self.df.y_pos = self.df.y_pos.fillna(method="pad")
        return self.df

    ###################################################################################################################
    # Run cointegration based on rolling window. If p_value is larger than confidence, do not take any position.
    # Apply after long_short_spread has been established
    def cointegration_control(self, coint_confidence, rolling_window):
        """ If coint_p_value is larger than coint_confidence, do not trade"""
        x_ret, y_ret = self.x.pct_change().dropna(), self.y.pct_change().dropna()
        A = pd.Series(y_ret.values, name='A')
        B = pd.Series(x_ret.values, name='B')
        coint_df = pd.concat([A, B], axis=1)
        coint_df['ii'] = range(len(coint_df))
        rolling_coint_pvalue = coint_df['ii'].rolling(rolling_window).apply(lambda ii: coint(coint_df.loc[ii, 'A'],
                                                                                             coint_df.loc[ii, 'B'])[1])
        rolling_coint_pvalue = pd.Series(rolling_coint_pvalue.values, index=x_ret.index, name="coint")
        self.df = pd.concat([self.df, rolling_coint_pvalue], axis=1)
        self.df.loc[self.df.coint > coint_confidence, "long_short_spread"] = 0


def kalman_filter_regression(x, y):
    # Transition Covariance
    trans_cov = np.array([[1e-4, 0.],
                          [0., 1e-6]])

    # Observation Matrix
    obs_mat = np.vstack([x, np.ones(x.shape)]).T[:, np.newaxis]

    kf = KalmanFilter(n_dim_obs=1,  # one observed value
                      n_dim_state=2,  # two states: slope and intercept
                      initial_state_mean=[0, 0],  # initiate means
                      initial_state_covariance=np.ones((2, 2)),  # initiate state covariances
                      transition_matrices=np.eye(2),  # identitiy matrix
                      observation_matrices=obs_mat,
                      observation_covariance=2,  # variance of y
                      transition_covariance=trans_cov)  # variance of coefficients

    state_means, state_covs = kf.filter(y.values)
    return state_means, state_covs


def kalman_filter_average(x):
    """ Kalman noise filtering on single series to extract hidden state."""
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=2,
                      observation_covariance=0.01,
                      transition_covariance=0.01)

    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    return state_means
