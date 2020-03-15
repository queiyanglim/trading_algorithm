import pandas as pd


def get_bbg_data():
    """ Daily prices since 1990"""
    path = "https://github.com/queiyanglim/trading_algorithm/raw/master/oil_trading/data/oil_prices.csv"

    df_pull = pd.read_csv(path, header=[0], index_col = 0)
    df_pull = df_pull[["CO1 Comdty", "CL1 Comdty"]]
    df_pull.index.name = "timestamp"
    df_pull = df_pull.rename(columns = {"CO1 Comdty": "brent",
                                        "CL1 Comdty": "wti"})
    df_pull.index = pd.to_datetime(df_pull.index, format = "%d/%m/%Y")
    df = df_pull.copy()
    df["spread"] = df.brent - df.wti
    # df = df.tail(2000)
    # df = np.log(df).diff()
    df = df.dropna()
    return df