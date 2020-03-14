from oil_trading.data.prepare_reuters_data import get_reuters_data
from oil_trading.brent_wti_kalman_spread.kalman_signal.trading_signal import kalman_regression_ZScore_signal
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

mpl.rcParams['figure.figsize'] = (10, 5)
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.linewidth'] = 0.01
plt.style.use("seaborn-whitegrid")

data = get_reuters_data("daily")

signal = kalman_regression_ZScore_signal(data, "wti", "brent")
signal.to_csv("trade_log.csv")

