from oil_trading.data.prepare_reuters_data import get_reuters_data
from oil_trading.data.prepare_bloomberg_data import get_bbg_data
from oil_trading.brent_wti_kalman_spread.kalman_signal.trading_signal import *
from pnl_process.static_position import trade_summary
from pnl_process.periodic_settlement import periodic_settlement
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

mpl.rcParams['figure.figsize'] = (10, 5)
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.linewidth'] = 0.01
plt.style.use("seaborn-whitegrid")

data = get_reuters_data("minute")
# data = get_bbg_data()
# data = data.tail(4*252)
capital = 100000

signal = kalman_regression_ZScore_signal(data, "wti", "brent", rolling_window=10, EM_on= False, EM_n_iter=5)
# df_units = static_unit_allocation(signal, capital)
df_units = dynamic_unit_allocation(signal, capital)

df_y = df_units[["y", "y_pos"]]
df_x = df_units[["x", "x_pos"]]

# _, pnl_x = trade_summary(df_x, "x", "x_pos")
# _, pnl_y = trade_summary(df_y, "y", "y_pos")

pnl_x = periodic_settlement(df_x, "x", "x_pos")
pnl_y = periodic_settlement(df_y, "y", "y_pos")
total_pnl = pnl_x + pnl_y
total_pnl.name = "total_pnl"

# Process Strategy Performance and plot performance
total_pnl.plot()
plt.show()
total_pnl = total_pnl.cumsum() + capital
total_pnl.plot()
plt.show()

# Trade Log
log = pd.concat([signal, df_units, pnl_x, pnl_y, total_pnl], axis = 1)
daily_ret = total_pnl.pct_change()
sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)

# Print trade log
# log.to_csv("trade_log.csv")
