import numpy as np
import matplotlib.pyplot as plt


def plot_buy_sell_signal_from_log(trade_log, spread_type="hedged_spread", _figsize=(20, 10)):
    """ spread_type = "mkt_spread" or "hedged_spread" """

    to_plot = trade_log[["x", "y", "hedged_spread", "y_pos"]].copy()
    to_plot = to_plot.loc[:, ~to_plot.columns.duplicated()]
    to_plot["mkt_spread"] = to_plot.y - to_plot.x
    to_plot["pos_chg"] = np.sign(to_plot.y_pos) != np.sign(to_plot.y_pos.shift(1))

    buy_signal = (to_plot.pos_chg) & (np.sign(to_plot.y_pos) > 0)
    short_signal = (to_plot.pos_chg) & (np.sign(to_plot.y_pos) < 0)
    close_signal = (to_plot.pos_chg) & (np.sign(to_plot.y_pos) == 0)

    to_plot.loc[buy_signal, "buy_sell_close"] = "buy"
    to_plot.loc[short_signal, "buy_sell_close"] = "sell"
    to_plot.loc[close_signal, "buy_sell_close"] = "close"
    to_plot.buy_sell_close = to_plot.buy_sell_close.fillna(0)

    style_buy = dict(size=12, color='green')
    style_sell = dict(size=12, color='red')
    style_close = dict(size=12, color='gray')
    style_dict = {"buy": style_buy, "sell": style_sell, "close": style_close}

    signal_plot = to_plot["hedged_spread"]
    ax = signal_plot.plot(figsize=_figsize, legend=True, linewidth=0.75, color="black", title="Kalman's Hedged Spread")

    for row in to_plot.iterrows():
        s = row[1]
        if s.buy_sell_close != 0:
            ax.text(s.name, s[signal_plot.name], s.buy_sell_close, fontdict=style_dict.get(s.buy_sell_close))
    plt.show()

