# encoding=utf-8

"""
"""
from matplotlib.pyplot import subplots
from data_source.stk_data_class import StkData
from pylab import plt


class GenStkPic(StkData):
    def __init__(self, df):
        super().__init__(stk_code=None)
        self.data = df

    def _plot_macd(self, ax, label):
        df = self.data
        ax.bar(range(0, len(df)), df['MACDhist'], label='MACD_' + label)
        ax.plot(range(0, len(df)), df['MACDsignal'], 'g-', linewidth=1)
        ax.plot(range(0, len(df)), df['MACD'], 'y-', linewidth=1)

        return ax

    def _plot_m_line(self, ax, m1=5, m2=30, m3=120):

        self.add_mean_col(col='close', m=m1)
        self.add_mean_col(col='close', m=m2)
        self.add_mean_col(col='close', m=m3)

        # fig, ax = subplots(ncols=1, nrows=1)

        ax.plot(range(0, len(self.data['date'])), self.data['close'], 'g*--', label='close')

        ax.plot(range(0, len(self.data['date'])), self.data['close_m%s' % str(m1)], 'b--', label='m%s' % str(m1))
        ax.plot(range(0, len(self.data['date'])), self.data['close_m%s' % str(m2)], 'r--', label='m%s' % str(m2))
        ax.plot(range(0, len(self.data['date'])), self.data['close_m%s' % str(m3)], 'y--', label='m%s' % str(m3))

        ax.set_title('%s~%s' % (str(self.data.head(1)['datetime'].values[0])[:16], str(self.data.tail(1)['datetime'].values[0])[:16]))

        return ax

    def plot_macd(self):

        fig, ax = subplots(ncols=1, nrows=2, sharex=True)
        ax[0] = self._plot_m_line(ax[0])
        ax[1] = self._plot_macd(ax[1], label='')
        plt.tight_layout(h_pad=0)

        for _ax in ax:
            _ax.legend(loc='best')

        return fig

