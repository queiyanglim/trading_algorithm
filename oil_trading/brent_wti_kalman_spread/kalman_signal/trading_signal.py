from oil_trading.brent_wti_kalman_spread.kalman_signal.kalman_filter import *


def kalman_regression_ZScore_signal(df_input, x_name, y_name, rolling_window, EM_on, EM_n_iter):
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

    # Negative sign to indicate opposite buy/sell direction between x and y
    df["hr"] = - state_means[:, 0]
    df["hedged_spread"] = df.y + df.hr * df.x

    # TODO: Z score parameters
    entry_zscore, exit_zscore = 1.5, 0.2
    mean_spread = df.hedged_spread.rolling(window=rolling_window).mean()
    std_spread = df.hedged_spread.rolling(window=rolling_window).std()
    df["z_score"] = (df.hedged_spread - mean_spread) / std_spread

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
    # df["num_units_long"].iloc[0] = 0
    # df["num_units_short"].iloc[0] = 0
    #
    # Forward filling
    df["num_units_long"] = df["num_units_long"].fillna(method="pad")
    df["num_units_short"] = df["num_units_short"].fillna(method="pad")
    df["long_short_spread"] = df.num_units_short + df.num_units_long
    return df


def kalman_regression_static_unit_allocation(df_with_signal, initial_capital=100000):
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
    df.loc[(df.long_short_spread == 0), ["x_pos", "y_pos"]] = 0

    df.loc[df.enter, "y_pos"] = df.long_short_spread * initial_capital / df.y.loc[df.enter]
    df.loc[df.enter, "x_pos"] = df.y_pos * df.hr
    df.x_pos = df.x_pos.fillna(method="pad")
    df.y_pos = df.y_pos.fillna(method="pad")
    return df


def dynamic_unit_allocation(df_with_signal, initial_capital=100000):
    df = df_with_signal.copy()
    df["y_pos"] = df.long_short_spread * initial_capital / df.y
    df["x_pos"] = df.hr * df.y_pos
    return df


def static_unit_allocation(df_with_signal, initial_capital=100000):
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
    df = df.iloc[40:]
    df.loc[(df.long_short_spread == 0), ["x_pos", "y_pos"]] = 0

    df.loc[df.enter, "y_pos"] = df.long_short_spread * initial_capital / df.y.loc[df.enter]
    df.loc[df.enter, "x_pos"] = df.y_pos * df.hr
    df.x_pos = df.x_pos.fillna(method="pad")
    df.y_pos = df.y_pos.fillna(method="pad")
    return df
