import pandas as pd


def periodic_settlement(df_input, security_name, security_pos):
    df = df_input.copy()
    # df = df.set_index("timestamp")
    df["realized_pnl"] = df[security_pos].shift(1) * (df[security_name] - df[security_name].shift(1))
    return df.realized_pnl
