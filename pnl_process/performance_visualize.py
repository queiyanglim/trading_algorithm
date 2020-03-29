from pnl_process import performance_statistics


class PerformancePlot(performance_statistics):
    def __init__(self, pnl_vector, risk_free_rate):
        super(PerformancePlot, self).__init__(pnl_vector, risk_free_rate)