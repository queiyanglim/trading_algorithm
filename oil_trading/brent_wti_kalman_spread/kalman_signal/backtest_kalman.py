from oil_trading.data.prepare_reuters_data import get_reuters_data
from oil_trading.data.prepare_bloomberg_data import get_bbg_data
from oil_trading.brent_wti_kalman_spread.kalman_signal.trading_signal import *
from pnl_process.performance_statistics import PerformanceStatistics
from pnl_process.periodic_settlement import periodic_settlement
import matplotlib.pyplot as plt
import matplotlib as mpl
from oil_trading.brent_wti_kalman_spread.kalman_signal.plotting_tool import *
from datetime import datetime
from scipy import stats

mpl.rcParams['figure.figsize'] = (10, 5)
mpl.rcParams['lines.linewidth'] = 0.75
plt.style.use("seaborn-whitegrid")

frequency = "hour"
data = get_reuters_data(frequency)

# if data is not daily price, filter data in between trading hours
# if frequency != "daily":
#     data = data.between_time("07:00", "18:00")

print("Data Shape", data.shape)
# data = get_bbg_data()
capital = 100000

signal, spread_data = kalman_regression_ZScore_signal(data, "wti", "brent",
                                                      entry_zscore=1.28,
                                                      exit_zscore=1.28,
                                                      rolling_window=10,
                                                      warm_up=0,
                                                      coint_control= False,
                                                      EM_on=True,
                                                      EM_n_iter=5)
# df_units = static_unit_allocation(signal, capital)
df_units = dynamic_unit_allocation(signal, capital)

df_y = df_units[["y", "y_pos"]]
df_x = df_units[["x", "x_pos"]]

pnl_x = periodic_settlement(df_x, "x", "x_pos")
pnl_y = periodic_settlement(df_y, "y", "y_pos")
total_pnl = pnl_x + pnl_y
total_pnl.name = "total_pnl"

# Trade Log
log = pd.concat([signal, df_units, total_pnl], axis=1)
# Print trade log
# log.to_csv("trade_log.csv")

performance = PerformanceStatistics(total_pnl, 0.0125, capital)
performance.print_result()
performance.plot_equity_chart()
performance.plot_normalized_equity_benchmark_chart()


# --------------------PLOT DATA------------------------------------------------------------
# Start and end date to observe trade data
fig, axes = plt.subplots(nrows=2)
fig.set_size_inches(20, 10.5)

# start_date = "2015-01-02"
# end_date = "2018-12-31"

start_date = log.index[-300]
end_date = log.index[-1]
plot_buy_sell_signal_from_log(axes[0], log.loc[start_date:end_date, :], spread_type="hedged_spread")

# plot z_score
spread_mean_std = spread_data[0].loc[start_date:end_date, :]
spread_mean_std.plot(ax=axes[0], title="Mean spread and Std Dev")
signal.loc[start_date:end_date].z_score.plot(ax=axes[1], title="Z-Score Signal")
plt.axhline(spread_data[1], color="green")
plt.axhline(-spread_data[1], color="green")
plt.axhline(spread_data[2], color="red")
plt.axhline(-spread_data[2], color="red")
plt.show()
