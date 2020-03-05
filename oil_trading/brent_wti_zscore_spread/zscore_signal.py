import pandas as pd


def rolling_zscore_trading_rule(df_with_signal):
    df = df_with_signal.copy()
    position = []
    pos_cache = 0  # cache position info before appending
    threshold = 1.0
    clear_level = 0.75
    # TODO: MODIFY TRADING RULES HERE
    for i, data in df.iterrows():
        if data.signal < -threshold:
            pos_cache = 1  # Long spread if spread's zscore is less than -threshold
        elif data.signal > threshold:
            pos_cache = -1  # short spread if spread's zscore is more than threshold
        elif abs(data.signal) < clear_level:
            pos_cache = 0
        position.append({"timestamp": data.name, "position": pos_cache})
    df_pos = pd.DataFrame(position).set_index("timestamp")
    return pd.concat([df, df_pos], axis=1)

# spread between y - X
def rolling_zscore_signal(df_input, X_name, y_name, zscore_window):
    df = df_input.copy()
    df["spread"] = df[y_name] - df[X_name]
    df["signal"] = rolling_zscore(df.spread, zscore_window)
    return df.dropna()


# https://stackoverflow.com/questions/47164950/compute-rolling-z-score-in-pandas-dataframe
def rolling_zscore(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x - m) / s
    return z
