from oil_trading.brent_wti_kalman_spread.kalman_signal.KalmanZScoreSignal import KalmanFilterZScoreSignal
from oil_trading.data.prepare_reuters_data import *
from oil_trading.data.prepare_reuters_data import get_reuters_data
from oil_trading.data.prepare_bloomberg_data import get_bbg_data
from oil_trading.brent_wti_kalman_spread.kalman_signal.KalmanOneFactor import KalmanOneFactor
from oil_trading.brent_wti_kalman_spread.kalman_signal.trading_signal import *
from pnl_process.performance_statistics import PerformanceStatistics
from pnl_process.periodic_settlement import periodic_settlement
import matplotlib.pyplot as plt
import matplotlib as mpl
from oil_trading.brent_wti_kalman_spread.kalman_signal.plotting_tool import *
from datetime import datetime
from scipy import stats
mpl.rcParams['figure.figsize'] = (7, 7)
mpl.rcParams['lines.linewidth'] = 0.75
plt.style.use("seaborn-whitegrid")

data = get_reuters_data("daily")

strategy = KalmanOneFactor(100000, data, "wti", "brent", -1, 1, 100)
df = strategy.backtest()
strategy.df.hedged_spread.plot()
plt.show()
strategy.df.long_short_spread.plot(marker = "x")
plt.show()

performance = PerformanceStatistics(strategy.pnl_vector, 0, 100000)
performance.print_result()
performance.plot_equity_chart()
performance.plot_normalized_equity_benchmark_chart()









