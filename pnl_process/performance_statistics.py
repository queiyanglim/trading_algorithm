from oil_trading.data.prepare_reuters_data import *
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


class PerformanceStatistics:
    """ Pnl vector must refer to a dataframe with daily p&L indexed with timestamps"""

    def __init__(self, pnl_vector, risk_free_rate, initial_capital):
        self.pnl_vector = convert_pnl_vector_to_daily(pnl_vector)
        self.initial_capital = initial_capital
        self.equity_value_series = (self.pnl_vector.cumsum() + initial_capital).dropna()
        self.equity_value_series.name = "equity_value"
        self.daily_equity_value = self.equity_value_series  # Convert into daily series
        self.daily_benchmark = get_daily_spx().reindex(self.daily_equity_value.index)  # Match same index as pnl series
        self.risk_free_rate = risk_free_rate
        self.annual_performance = annual_performance(self.daily_equity_value)
        self.annual_std_dev = annual_std_dev(self.daily_equity_value)
        self.max_drawdown_percent = max_drawdown_percent(self.daily_equity_value)
        self.max_drawdown_value = max_drawdown_value(self.daily_equity_value)
        self.compounding_annual_performance = compounding_annual_performance(
            initial_capital=self.equity_value_series.iloc[0],
            final_capital=self.equity_value_series.iloc[-1],
            num_of_trading_days=len(
                self.daily_equity_value.index)
            )
        self.beta = beta(self.daily_equity_value, self.daily_benchmark)
        self.alpha = alpha(self.daily_equity_value, self.daily_benchmark, self.risk_free_rate)
        self.sharpe_ratio = sharpe_ratio(self.daily_equity_value, self.risk_free_rate)
        self.treynor_ratio = treynor_ratio(self.daily_equity_value, self.daily_benchmark, self.risk_free_rate)

    def result(self):
        res = { "Risk Free Rate": self.risk_free_rate,
                "Annual Performance": self.annual_performance,
                "Annual Standard Deviation": self.annual_std_dev,
                "Max Drawdown": self.max_drawdown_percent,
                "Compounding Annual Performance": self.compounding_annual_performance,
                "Beta": self.beta,
                "Alpha": self.alpha,
                "Sharpe Ratio": self.sharpe_ratio,
                "Treynor Ratio": self.treynor_ratio
                }
        return res

    def print_result(self):
        res = self.result()
        for r in res:
            print(r, ": ", res.get(r))

    def plot_equity_chart(self):
        self.daily_equity_value.plot(title=f"Equity Chart")
        plt.show()

    def plot_normalized_equity_benchmark_chart(self):
        plot_df = pd.concat([self.daily_benchmark, self.daily_equity_value], axis=1)
        plot_df = plot_df.dropna()
        plot_df = plot_df / plot_df.iloc[0]
        plot_df.plot(title=f"Normalized Equity Value vs Benchmark: {self.daily_benchmark.name}")
        plt.show()


def convert_pnl_vector_to_daily(pnl_df):
    """" In case of trading in higher frequency than daily, convert pnl series into daily price series"""
    df = pnl_df.copy()
    # Smart guess data frequency (in days)
    freq_in_days = float(stats.mode((np.diff(df.index) / np.timedelta64(1, 'D')))[0])
    # If frequency is not in days, regroup pnl
    if freq_in_days != 1:
        return df.resample("D").sum()
    else:
        return df


def annual_performance(daily_equity_value, trading_days_per_year=252):
    """" Return annual performance of strategy given daily pnl series """
    daily_equity_return = daily_equity_value.pct_change().dropna()
    return (daily_equity_return.mean() + 1) ** trading_days_per_year - 1.0


def annual_variance(daily_equity_value, trading_days_per_year=252):
    """" Return annual variance of strategy given daily pnl series """
    daily_equity_return = daily_equity_value.pct_change().dropna()
    return daily_equity_return.var() * trading_days_per_year


def annual_std_dev(daily_equity_value, trading_days_per_year=252):
    return np.sqrt(annual_variance(daily_equity_value, trading_days_per_year))


def max_drawdown_percent(daily_equity_value, num_to_show=5):
    """Show, by default, 5 worst loss dates and their largest pct drop"""
    df = daily_equity_value.copy()
    df = df.pct_change()
    df = df.dropna()
    return df.sort_values(ascending=True)[:num_to_show]


def max_drawdown_value(daily_equity_value, num_to_show=5):
    """Show, by default, 5 worst loss dates and their losses"""
    df = daily_equity_value.copy()
    df = df.diff()
    df = df.dropna()
    return df.sort_values(ascending=True)[:num_to_show]


def compounding_annual_performance(final_capital, initial_capital, num_of_trading_days, trading_days_per_year=252):
    year = num_of_trading_days / trading_days_per_year
    return (final_capital / initial_capital) ** (1 / year) - 1


# Covariance between the algorithm and benchmark performance, divided by benchmark's variance
def beta(daily_equity_value, daily_benchmark_series):
    """"
    Calculate beta of strategy against chosen benchmark
    """
    concat_df = pd.concat([daily_equity_value, daily_benchmark_series], axis=1)
    concat_df_ret = concat_df.pct_change().dropna()
    cov_mat = concat_df_ret.cov()
    return cov_mat.iloc[0, 1] / cov_mat.iloc[1, 1]


# Abnormal returns over the risk free rate and the relationshio (beta) with the benchmark returns.
def alpha(daily_equity_value, daily_benchmark_series, risk_free_rate):
    return annual_performance(daily_equity_value) - \
           (risk_free_rate + beta(daily_equity_value, daily_benchmark_series)) * \
           (annual_performance(daily_benchmark_series) - risk_free_rate)


def sharpe_ratio(daily_equity_value, risk_free_rate):
    return (annual_performance(daily_equity_value) - risk_free_rate) / annual_std_dev(daily_equity_value)


def treynor_ratio(daily_equity_value, daily_benchmark_series, risk_free_rate):
    return (annual_performance(daily_equity_value) - risk_free_rate) / beta(daily_equity_value, daily_benchmark_series)
