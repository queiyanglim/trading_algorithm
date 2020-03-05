from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


def equity_performance(df_with_position, X_name, y_name):
    df = df_with_position.copy()
    long_short, X_pos, y_pos = 0, 0, 0
    capital = 100000

    position_log = []
    # Spread trading y_name - X_name
    # TODO: MODIFY TRADING RULES HERE
    for i, data in df.iterrows():
        # if portfolio is not currently long
        if long_short != 1:
            # if signal suggest going long
            if data.position == 1:
                long_short = 1
                y_pos = np.floor(capital / data[y_name])
                X_pos = -np.floor(capital / data[X_name])

        # if portfolio is previously long
        else:
            # if signal suggest going short
            if data.position == -1:
                long_short = -1
                y_pos = -np.floor(capital / data[y_name])
                X_pos = np.floor(capital / data[X_name])
        position_log.append({"timestamp": data.name, "X_pos": X_pos, "y_pos": y_pos})

    df_trade = pd.DataFrame(position_log).set_index("timestamp")

    df_equity = pd.concat([df, df_trade], axis=1)
    df_equity["equity"] = df_equity["X_pos"] * df_equity[X_name] + df_equity["y_pos"] * df_equity[y_name]

    return df_equity


# Parse df with "signal"
# Framework for parsing signals
def rolling_regression_trading_rule(df_with_signal):
    df = df_with_signal.copy()
    position = []
    for i, data in df.iterrows():
        if data.spread > data.signal:
            position.append({"timestamp": data.name, "position": 1})
        elif data.spread < data.signal:
            position.append({"timestamp": data.name, "position": -1})
        else:
            position.append({"timestamp": data.name, "position": 0})
    # df_pos = pd.DataFrame()
    return position


# BACK TEST ONLY
def rolling_regression_trading_signal(df_input, X_name, y_name, update_window, ewm_span, fit_intercept=True):
    df_signal = rolling_regression(df_input, X_name, y_name, update_window, fit_intercept=fit_intercept)

    fair_spread = (df_signal.alpha - 1) * df_signal[X_name] + df_signal.beta
    fair_spread = pd.DataFrame(fair_spread.rename("fair_spread", inplace=True))

    df_spread = pd.concat([df_signal, fair_spread], axis=1)
    df_spread["signal"] = df_spread.fair_spread.ewm(span=ewm_span).mean()
    df_spread["spread"] = df_spread[y_name] - df_spread[X_name]
    return df_spread


# BACK TEST ONLY
def rolling_regression(df_input, X_name, y_name, update_window, fit_intercept=True):
    alpha_list, beta_list, r_2 = [], [], []
    for i in range(update_window, len(df_input)):
        rolled_df = df_input[i - update_window + 1: i + 1]

        X = rolled_df[X_name].values.reshape(-1, 1)
        y = rolled_df[y_name].values

        lin_reg = LinearRegression(fit_intercept=fit_intercept)
        lin_reg.fit(X, y)
        alpha, beta = lin_reg.coef_, lin_reg.intercept_
        alpha_list.append(float(alpha))
        beta_list.append(float(beta))
        r_2.append(float(lin_reg.score(X, y)))

    df_with_coef_and_intercept = df_input.copy()[update_window:]
    df_with_coef_and_intercept["alpha"] = alpha_list
    df_with_coef_and_intercept["beta"] = beta_list
    df_with_coef_and_intercept["r_squared"] = r_2

    return df_with_coef_and_intercept
