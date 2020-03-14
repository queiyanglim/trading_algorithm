import pandas as pd


def trade_summary(df_input, security_name, position_name):
    """
    For static positions only, i.e. at any time a fixed unit of positions are live.
    Take a dataframe with timestamp index, security, and security_pos (position) and calculate PnL trade by trade.
    """
    df = df_input.copy()
    df["long_short"] = (df[position_name] > 0) * 1 - (df[position_name] < 0) * 1

    trade_detail = []

    def update_trade(_trade_count, _position, _open_date, _open_price, _close_date, _close_price):
        trade_detail.append({"trade": _trade_count,
                             "position": _position,
                             "open_date": _open_date,
                             "open_price": _open_price,
                             "close_date": _close_date,
                             "close_price": _close_price,
                             "realized_pnl": _position * (_close_price - open_price)})

    trade_count = 0
    long_short = 0

    for i, data_slice in enumerate(df.iterrows()):
        s = data_slice[1]  # Slice
        if i > 1 and s.long_short != df.iloc[i - 1].long_short:
            if long_short != 0:
                close_price, close_date = s[security_name], s.name
                update_trade(trade_count, position, open_date, open_price, close_date, close_price)
                long_short = 0

            if s.long_short != 0:
                open_price = s[security_name]
                position = s[position_name]
                open_date = s.name  # date/time from index
                trade_count += 1
                long_short = s.long_short

        if s.long_short != long_short:
            close_price, close_date = s[security_name], s.name
            close_date = s.name
            update_trade(trade_count, position, open_date, open_price, close_date, close_price)

    trade_summary_df = pd.DataFrame(trade_detail)

    # Merge realized PnL onto original time_series. TODO: Can consider returning only one single series
    trade_time_series = trade_summary_df[["close_date", "realized_pnl"]]
    trade_time_series = trade_time_series.set_index("close_date")
    trade_time_series.index.name = df_input.index.name
    # TODO: AMEND DATETIME FORMAT WHEN NECESSARY
    trade_time_series.index = pd.to_datetime(trade_time_series.index, format="%d/%m/%Y")
    trade_time_series = pd.concat([trade_time_series, df], axis=1)
    trade_time_series.realized_pnl = trade_time_series.realized_pnl.fillna(0)

    return trade_summary_df, trade_time_series.realized_pnl


def _test_static_trade_summary():
    # trade = pd.read_csv(r"C:\Users\queiy\trading_algorithm\pnl_process\trade_sample.csv")
    trade = trade.set_index("timestamp")
    trade = trade.fillna(0)
    trade.index = pd.to_datetime(trade.index, format="%d/%m/%Y")

    x, y = trade_summary(trade, "y", "y_pos")
    pass
