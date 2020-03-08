# import numpy as np
# import pandas as pd
#
#
# def equity_performance(df_with_position, X_name, y_name, capital = 100000):
#     df = df_with_position.copy()
#     long_short, X_pos, y_pos = 0, 0, 0
#     position_log = []
#
#     # Spread trading y_name - X_name
#     for i, data in df.iterrows():
#         # On LONG signal
#         if data.position == 1:
#             if long_short != 1:
#                 long_short = 1
#                 y_pos = np.floor(capital / data[y_name])
#                 X_pos = -np.floor(capital / data[X_name])
#         # On SHORT signal
#         elif data.position == -1:
#             if long_short != -1:
#                 long_short = -1
#                 y_pos = -np.floor(capital / data[y_name])
#                 X_pos = np.floor(capital / data[X_name])
#         else:
#             long_short = 0
#             y_pos = 0
#             X_pos = 0
#         position_log.append({"timestamp": data.name,
#                              "X_pos": X_pos, "y_pos": y_pos})
#
#     df_trade = pd.DataFrame(position_log).set_index("timestamp")
#     df_equity = pd.concat([df, df_trade], axis=1)
#     # df_equity["daily_mtm"] = df_equity["X_pos"] * df_equity[X_name] + df_equity["y_pos"] * df_equity[y_name]
#     # df_equity["equity"] = df_equity["daily_mtm"].cumsum() + capital
#
#     # Performance
#     ret = df_equity.equity.pct_change().dropna()
#     sharpe_ratio = ret.mean() / ret.std()
#     performance = {"sharpe_ratio": sharpe_ratio,
#                    "max_drawdown": ret.min(),
#                    "win": sum(ret.values > 0),
#                    "lose": sum(ret.values < 0),
#                    "flat": sum(ret.values == 0),
#                    "winning_prob": sum(ret.values > 0)/len(ret.values)}
#
#     return df_equity, performance