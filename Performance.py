# !usr/bin/env/python3
# -*- encoding:utf-8 -*-
# @date     : 2020/02/04
# @author   : CHEN WEI
# @filename : Performance.py
# @software : pycharm

import os
import pandas as pd
import numpy as np
from dateutil.parser import parse


class Metrics:
    """
    Calculate some easy backtest indicators.
    :examples:
    >>> path = os.path.join(os.getcwd(), '2020012807.xlsx')
    >>> path = os.path.join(os.getcwd(), '2020020503.xlsx')
    >>> nav = pd.read_excel(path, index_col='date').iloc[:, -2]
    >>> nav_bench = pd.read_excel(path, index_col='date').iloc[:, -1]
    """

    @staticmethod
    def annual_return(nav):
        """
        Calculate annual return of given nav.
        :param nav: pd.Series.
        :examples:
        >>> Metrics.annual_return(nav)
        """
        ret = (nav[-1] / nav[0]) ** (252 / len(nav)) - 1
        return ret

    @staticmethod
    def annual_vol(nav):
        """
        Calculate annual volatility of given nav.
        :examples:
        >>> Metrics.annual_vol(nav)
        """
        vol = np.std(nav / nav.shift(1) - 1) * np.sqrt(252)
        return vol

    @staticmethod
    def sharpe_ratio(nav):
        """
        Calculate sharpe ratio of given nav.
        :examples:
        >>> Metrics.sharpe_ratio(nav)
        """
        sr = Metrics.annual_return(nav) / Metrics.annual_vol(nav)
        return sr

    @staticmethod
    def max_drawdown(nav):
        """

        :examples:
        >>> Metrics.max_drawdown(nav)
        """
        drawdown = []
        for i, value in enumerate(nav):
            drawdown.append(1 - (value / max(nav[:(i+1)])))
        return max(drawdown)

    @staticmethod
    def calmar_ratio(nav):
        """

        :examples:
        >>> Metrics.calmar_ratio(nav)
        """
        calmar_ratio = Metrics.annual_return(nav) / Metrics.max_drawdown(nav)
        return calmar_ratio

    @staticmethod
    def excess_return(nav, nav_bench):
        """

        :param nav_bench: pd.Series. Nav series of benchmark.
        :examples:
        >>> Metrics.excess_return(nav, nav_bench)
        """
        ra = nav / nav.shift(1) - 1
        rb = nav_bench / nav_bench.shift(1) - 1
        excess_nav = (1 + ra - rb).cumprod()
        excess_value = Metrics.annual_return(excess_nav.dropna())

        return excess_value

    @staticmethod
    def excess_vol(nav, nav_bench):
        """
        :examples:
        >>> Metrics.excess_return(nav, nav_bench)
        """
        ra = nav / nav.shift(1) - 1
        rb = nav_bench / nav_bench.shift(1) - 1
        excess_nav = (1 + ra - rb).cumprod()
        excess_vol = Metrics.annual_vol(excess_nav)

        return excess_vol

    @staticmethod
    def info_ratio(nav, nav_bench):
        """
        :examples:
        >>> Metrics.info_ratio(nav, nav_bench)
        """
        info_ratio = Metrics.excess_return(nav, nav_bench) / Metrics.excess_vol(nav, nav_bench)
        return info_ratio

    @staticmethod
    def excess_max_drawdown(nav, nav_bench):
        """
        :examples:
        >>> Metrics.excess_max_drawdown(nav, nav_bench)
        """
        ra = nav / nav.shift(1) - 1
        rb = nav_bench / nav_bench.shift(1) - 1
        excess_nav = (1 + ra - rb).cumprod().dropna()

        return Metrics.max_drawdown(excess_nav)

    @staticmethod
    def single_performance(nav, nav_bench):
        """

        :example:
        >>> Metrics.single_performance(nav, nav_bench)
        """
        if nav.index[0].year == nav.index[-1].year:
            year = nav.index[0].year
        else:
            year = '成立以来'

        performance = {
            '时间': year,
            '年化收益率': Metrics.annual_return(nav),
            '年化波动率': Metrics.annual_vol(nav),
            '夏普比率': Metrics.sharpe_ratio(nav),
            '最大回撤': Metrics.max_drawdown(nav),
            '卡玛比率': Metrics.calmar_ratio(nav),
            '年化超额收益': Metrics.excess_return(nav, nav_bench),
            '超额收益年化波动率': Metrics.excess_vol(nav, nav_bench),
            '信息比率': Metrics.info_ratio(nav, nav_bench),
            '超额收益最大回撤': Metrics.excess_max_drawdown(nav, nav_bench)
        }
        return performance


class Return:
    def __init__(self, nav, nav_bench):
        self.nav = nav
        self.nav_bench = nav_bench
        self.nav_df = pd.DataFrame({'nav': self.nav, 'nav_bench': self.nav_bench})
        self.attr = ['时间', '年化收益率', '年化波动率', '夏普比率', '最大回撤', '卡玛比率', '年化超额收益',
                     '超额收益年化波动率', '信息比率', '超额收益最大回撤']

    def performance(self):
        """
        Calculate some simple indicators during different years of give nav.
        :return: pd.DataFrame
        :example:
        >>> re = Return(nav, nav_bench)
        >>> pe = re.performance()
        """
        perform = []
        for year, _nav in self.nav_df.groupby(lambda x: x.year):
            perform.append(Metrics.single_performance(_nav['nav'], _nav['nav_bench']))

        perform.append(Metrics.single_performance(self.nav_df['nav'], self.nav_df['nav_bench']))
        return pd.DataFrame(perform)[self.attr]


def cal_multi_performances(path, index_col=None):
    """

    :param path: dict. key of dict is code of nav, value of dict is file path.
    :param index_col: Any.
    :return: pd.DataFrame. performances of multiple nav.
    """
    results = pd.DataFrame()
    for key, value in path.items():
        data = pd.read_excel(value, index_col=index_col)

        # if index not in form of datetime64, parse it.
        if isinstance(data.index[0], str):
            data.index = pd.Series(data.index).apply(parse)

        performance = Return(data.iloc[:, -2], data.iloc[:, -1]).performance()
        performance['编号'] = key
        results = pd.concat([results, performance])

    return results


if __name__ == '__main__':
    _dir = os.getcwd()

    # 计算滚动组的回测收益指标
    ro_codes = ['021201']
    ro_files = [_dir + '/saved_model/滚动/' + code + '/2020' + code + '滚动.xlsx' for code in ro_codes]
    ro_path = dict(zip(ro_codes, ro_files))

    ro_results = cal_multi_performances(ro_path)

    with pd.ExcelWriter(os.path.join(_dir, '滚动收益指标汇总.xlsx'), datetime_format='YYYY-MM-DD') as writer:
        for i, df in ro_results.groupby('编号'):
            df.to_excel(writer, index=False, sheet_name=df['编号'][0])

    # 图表18
    codes = ['021201conv+LSTM', '021208conv+GRU', '021209GRU', '021210LSTM']
    files = [_dir + '/saved_model/结果记录/' + code[:4] + '/' + code + '/2020' + code[:6] + '.xlsx' for code in codes]
    path = dict(zip(codes, files))

    results = cal_multi_performances(path, 'date')

    with pd.ExcelWriter(os.path.join(_dir, '收益指标汇总.xlsx'), datetime_format='YYYY-MM-DD') as writer:
        for i, df in results.groupby('编号'):
            df.to_excel(writer, index=False, sheet_name=df['编号'][0])


    # 图表22-41,中间这些表只是编号不一样，计算过程一致
    codes = ['021201', '021202', '021203', '021204', '021205', '021206', '021207']  # 图表22
    codes = ['020901', '020902', '020903', '020904', '020905', '021201', '021601', '021602', '021603']  # 图表25
    codes = ['021213', '021214', '021215', '021216', '021217', '021218', '021219']  # 图表28
    codes = ['021201', '021204', '021211', '021212']  # 图表31
    codes = ['021701', '021707']  # 图表35、36
    codes = ['021701', '021702', '021703', '021704', '021705', '021706']  # 图表38
    codes = ['022101']  # 图表44
    codes = ['022001', '021901']  # 图表49、50

    files = [_dir + '/saved_model/结果记录/' + code[:4] + '/' + code + 'conv+LSTM/2020' + code + '.xlsx' for code in codes]
    path = dict(zip(codes, files))
    results = cal_multi_performances(path, 'date')

    with pd.ExcelWriter(os.path.join(_dir, '收益指标汇总.xlsx'), datetime_format='YYYY-MM-DD') as writer:
        for i, df in results.groupby('编号'):
            df.to_excel(writer, index=False, sheet_name=df['编号'][0])

