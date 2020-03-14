from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS


def half_life(spread):
    """" Optimal rolling mean window based on half life of deviation from mean"""
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2 = add_constant(spread_lag)
    model = OLS(spread_ret, spread_lag2)
    res = model.fit()
    halflife = int(round(-np.log(2) / res.params[1], 0))

    if halflife <= 0:
        halflife = 1
    return halflife

