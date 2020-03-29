import os
import pandas as pd


def get_reuters_data(frequency):
    """ Import Reuters Data from Github based on frequency.
    Available frequency: "minute", "daily".
    """

    path_list = {
        "daily": "https://github.com/queiyanglim/trading_algorithm/blob/master/oil_trading/data/daily.csv?raw=true",
        "minute": "https://github.com/queiyanglim/trading_algorithm/blob/master/oil_trading/data/minute.csv?raw=true",
        "hour": "https://github.com/queiyanglim/trading_algorithm/blob/master/oil_trading/data/hour.csv?raw=true"}

    df_pull = pd.read_csv(path_list.get(frequency), header=[0, 1], index_col=0)
    df_pull.index = pd.to_datetime(df_pull.index, format="%Y-%m-%d")

    # Prepare Data
    brent = df_pull.brent.CLOSE
    brent.name = "brent"
    wti = df_pull.wti.CLOSE
    wti.name = "wti"

    # Concat the brent and wti series
    df_pull = pd.concat([brent, wti], axis=1)
    df_pull["spread"] = df_pull.brent - df_pull.wti

    # Cleaning NaN values
    df_pull = df_pull.dropna()
    return df_pull


def _test_get_reuters_data():
    print(get_reuters_data("daily"))
    print(get_reuters_data("minute"))
    pass