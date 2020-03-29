from oil_trading.brent_wti_kalman_spread.kalman_signal.kalman_filter import *
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.tsa.stattools import coint
from scipy import stats
import matplotlib.pyplot as plt

def kalman_regression_ZScore_signal(df_input, x_name, y_name, entry_zscore,
                                    exit_zscore, rolling_window, warm_up,
                                    coint_control, EM_on, EM_n_iter):
    """ Take a df with x and y (indexed with timestamp) and
    return a df with ZScore signal.
    *x_name to be called using df[x_name] for its series"""

    x, y = df_input[x_name], df_input[y_name]
    df = pd.DataFrame({"y": y, "x": x})
    df.index = pd.to_datetime(df_input.index)
    state_means = kalman_filter_regression(kalman_filter_average(x),
                                           kalman_filter_average(y),
                                           EM_on=EM_on,
                                           EM_n_iter=EM_n_iter)
    df["hr"] = - state_means[:, 0]
    df["hedged_spread"] = df.y + df.hr * df.x

    # ---------------------------------------------------------------------------------------------------------
    # TODO: VERSION 1 - (FIXED ROLLING WINDOW) Z score parameters
    mean_spread = df.hedged_spread.rolling(window=rolling_window).mean()
    std_spread = df.hedged_spread.rolling(window=rolling_window).std()
    mean_spread.name = "mean_spread"
    std_spread.name = "std_spread"
    df["z_score"] = (df.hedged_spread - mean_spread) / std_spread
    # ---------------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------------------
    # TODO: VERSION 2 - VOL CONTROLLED WINDOW
    # Fit a empirical CDF on historical volatility (not annualized)
    # vol_rolling_window = 5
    # vol_data = df.y - df.x
    # # vol_data = vol_data.pct_change().rolling(vol_rolling_window).std()
    # # vol_data = vol_data.pct_change().ewm(span=vol_rolling_window).std()
    # vol_data = vol_data.diff().ewm(span=vol_rolling_window).std()
    # vol_data = vol_data.diff().rolling(window=vol_rolling_window).std()
    # vol_ecdf = ECDF(vol_data.values)
    # fix_window = lambda x: np.floor((1 - vol_ecdf(x)) * 10 + 2)
    # df["aug_window"] = fix_window(vol_data)
    #
    # # Create a z-score dataframe containing all columns of possible windows
    # all_possible_windows = map(int, list(set(df.aug_window.values)))
    #
    # data = df.hedged_spread.copy()
    # dynamic_rolling_mean = pd.DataFrame(index=data.index)
    # dynamic_rolling_std = dynamic_rolling_mean.copy()
    #
    # for window in all_possible_windows:
    #     dynamic_rolling_mean[f"window_{window}"] = data.rolling(window=window).mean()
    #     dynamic_rolling_std[f"window_{window}"] = data.rolling(window=window).std()
    #
    # # dynamic_rolling_mean.tail(20).plot(figsize=(20,15))
    # for s in dynamic_rolling_mean.iterrows():
    #     roll_window = "window_" + str(int(df.loc[s[0], "aug_window"]))
    #     df.loc[s[0], "dy_roll_mean"] = s[1][roll_window]
    #
    # for s in dynamic_rolling_std.iterrows():
    #     roll_window = "window_" + str(int(df.loc[s[0], "aug_window"]))
    #     df.loc[s[0], "dy_roll_std"] = s[1][roll_window]
    #
    # df["z_score"] = (df.hedged_spread.values - df.dy_roll_mean.values) / df.dy_roll_std.values
    # mean_spread = df.dy_roll_mean
    # std_spread = df.dy_roll_std
    # ---------------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------------------
    # TODO: VERSION 3 - Chi2/Kolmogorov-Smirnov VOL DISTRIBUTION TO SAMPLE ROLLING WINDOW
    # Fit a distribution on historical volatility (annualized)

    # Annualized factor. If day then == 1, if hour then == 1/24 and so on.
    # ann_scale = float(stats.mode((np.diff(df.index) / np.timedelta64(1, 'D')))[0])
    # # vol_data = df.y.pct_change().rolling(window=10).std(ddof=1) * np.sqrt((1 / ann_scale) * 252)
    # vol_data = df.y.pct_change().ewm(span=3).std(ddof=1) * np.sqrt((1 / ann_scale) * 252)
    # vol_data = vol_data.fillna(0)
    #
    # # dist = stats.chi2
    # dist = stats.kstwobign
    # params = dist.fit(vol_data)
    # arg = params[:-2]
    # loc = params[-2]
    # scale = params[-1]
    #
    # pdf_x = lambda x: np.floor(dist.pdf(x, loc=loc, scale=scale, *arg) * scale * 8) + 1
    #
    # # Plot window profile
    # s = np.arange(0, vol_data.max(), 0.01)
    # plt.plot(s, pdf_x(s))
    # plt.title("Rolling Windoow Profile")
    # plt.show()
    # df["aug_window"] = pdf_x(vol_data)
    #
    # # Create a z-score dataframe containing all columns of possible windows
    # all_possible_windows = map(int, list(set(df.aug_window.values)))
    #
    # data = df.hedged_spread.copy()
    # dynamic_rolling_mean = pd.DataFrame(index=data.index)
    # dynamic_rolling_std = dynamic_rolling_mean.copy()
    #
    # for window in all_possible_windows:
    #     dynamic_rolling_mean[f"window_{window}"] = data.rolling(window=window).mean()
    #     dynamic_rolling_std[f"window_{window}"] = data.rolling(window=window).std()
    #
    # # dynamic_rolling_mean.tail(20).plot(figsize=(20,15))
    # for s in dynamic_rolling_mean.iterrows():
    #     roll_window = "window_" + str(int(df.loc[s[0], "aug_window"]))
    #     df.loc[s[0], "dy_roll_mean"] = s[1][roll_window]
    #
    # for s in dynamic_rolling_std.iterrows():
    #     roll_window = "window_" + str(int(df.loc[s[0], "aug_window"]))
    #     df.loc[s[0], "dy_roll_std"] = s[1][roll_window]
    #
    # df["z_score"] = (df.hedged_spread.values - df.dy_roll_mean.values) / df.dy_roll_std.values
    # mean_spread = df.dy_roll_mean
    # std_spread = df.dy_roll_std
    # ---------------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------------------


    # TODO: Trading Logic
    # Enter long position if current z-score is less than entry threshold AND previous day was within threshold bound
    # Exit long position if previous day was less than exit score and current score is larger than exit score
    df["long_entry"] = ((df.z_score < -entry_zscore) & (df.z_score.shift(1) > - entry_zscore))
    df["long_exit"] = ((df.z_score > -exit_zscore) & (df.z_score.shift(1) < - exit_zscore))

    # Enter short position if otherwise
    df["short_entry"] = ((df.z_score > entry_zscore) & (df.z_score.shift(1) < entry_zscore))
    df["short_exit"] = ((df.z_score < exit_zscore) & (df.z_score.shift(1) > exit_zscore))

    # Set up positions
    # FIXME: Simplify this
    df["num_units_long"] = np.nan
    df["num_units_short"] = np.nan
    df.loc[df.long_entry, "num_units_long"] = 1
    df.loc[df.long_exit, "num_units_long"] = 0
    df.loc[df.short_entry, "num_units_short"] = -1
    df.loc[df.short_exit, "num_units_short"] = 0
    df.loc[df.index[0], ["num_units_long", "num_units_short"]] = 0

    # Forward filling
    df["num_units_long"] = df["num_units_long"].fillna(method="pad")
    df["num_units_short"] = df["num_units_short"].fillna(method="pad")
    df["long_short_spread"] = df.num_units_short + df.num_units_long

    # warm up period
    df.loc[:warm_up, "long_short_spread"] = 0
    df = df.tail(len(df) - warm_up)


    # If coint p-value fails, do not trade!
    # TODO: COINTEGRATION CONTROL
    if coint_control is True:
        x_ret, y_ret = x.pct_change().dropna(), y.pct_change().dropna()
        A = pd.Series(y_ret.values, name='A')
        B = pd.Series(x_ret.values, name='B')

        coint_df = pd.concat([A, B], axis=1)
        coint_df['ii'] = range(len(coint_df))

        rolling_coint_pvalue = coint_df['ii'].rolling(warm_up).apply(lambda ii: coint(coint_df.loc[ii, 'A'], coint_df.loc[ii,
                                                                                                                     'B'])[1])
        rolling_coint_pvalue = pd.Series(rolling_coint_pvalue.values, index=x_ret.index, name = "coint")
        df = pd.concat([df, rolling_coint_pvalue], axis = 1)
        coint_confidence = 0.0025
        df.loc[df.coint > coint_confidence, "long_short_spread"] = 0
    # ---------------------------------------------------------------------------------------------------------

    return df, [pd.concat([mean_spread, std_spread], axis=1), entry_zscore, exit_zscore]


# Allocate fixed amount of capital at period
def dynamic_unit_allocation(df_with_signal, initial_capital):
    df = df_with_signal.copy()
    df["y_pos"] = np.floor(df.long_short_spread * initial_capital / df.y)
    df["x_pos"] = np.floor(df.hr * df.y_pos)
    return df


def static_unit_allocation(df_with_signal, initial_capital):
    """" Take df with signals containing:
        1. num_units_long
        2. num_units_short
        3. long_entry
        4. short_entry
        and return number of units to long/short based on initial capital
        as well as trade logs for x and y"""
    df = df_with_signal.copy()
    df["long_short_spread"] = df.num_units_long + df.num_units_short
    # df["enter"] = (df.long_entry & (df.long_short_spread == 1)) | (df.short_entry & (df.long_short_spread == -1))
    df["enter"] = (df.long_short_spread != df.long_short_spread.shift(1)) & ((df.short_entry) | (df.long_entry))
    df = df[["y", "x", "hr", "long_short_spread", "enter"]]

    df["x_pos"] = np.nan
    df["y_pos"] = np.nan
    df = df.iloc[40:]  # FIXME: may not be necessary
    df.loc[(df.long_short_spread == 0), ["x_pos", "y_pos"]] = 0

    df.loc[df.enter, "y_pos"] = df.long_short_spread * initial_capital / df.y.loc[df.enter]
    df.loc[df.enter, "x_pos"] = df.y_pos * df.hr
    df.x_pos = df.x_pos.fillna(method="pad")
    df.y_pos = df.y_pos.fillna(method="pad")
    return df
