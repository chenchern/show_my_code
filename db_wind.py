# -*- coding: utf-8 -*-
# @Time     : 2019/7/12 18:29
# @Author   ：Chen Wei
# @File     : db_wind.py
# @Software : PyCharm


from wm.lang_util import PdUtil, SaUtil, columns_from_args
from wm.return_metrics import Nav, MetricsMap
from wm.db_dict import Dict
from wm.db import DataBase
from wm.data_util import DateUtil
from wm.xts_util import Xts
from wm.data_util import DataUtil
from wm.decorator import parse_df_date

from sqlalchemy import Column, String, Numeric, DateTime, Table, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.query import Query
from sqlalchemy.orm import Load, load_only
from dateutil.parser import parse
import datetime
import re
import pandas as pd
import empyrical
from functools import partial


Base = declarative_base()
db_wind = DataBase('wind')
engine_wind = db_wind.db_engine()
sess_wind = db_wind.db_session()


"Part of the code written."

class DbWind:

    """
    Query API for  wind database.
    """

    @staticmethod
    def mf_manager():
        """
        Query china mutual fund manager
        Returns
        -------
        Examples
        ------
        >>> test = DataBase.query_to_df(DbWind.mf_manager().filter(MfManager.F_INFO_FUNDMANAGER.in_(['林鹏'])))
        """
        return sess_wind.query(MfManager)

    @staticmethod
    def mf_info():
        """
        Query ChinaMutualFundDescription.
        Returns
        -------

        """
        return sess_wind.query(MfInfo)

    @staticmethod
    def mf_share():
        """
        Query ChinaMutualFundShare.
        Returns
        -------

        """
        return sess_wind.query(MfShare)

    @staticmethod
    def mf_nav():
        """
        Query ChinaMutualFundNav.
        Returns
        -------

        """
        return sess_wind.query(MfNav)

    @staticmethod
    def mf_bm_nav():
        """
        Query ChinaMutualFundBenchmarkEOD.
        Returns
        -------

        """
        return sess_wind.query(MfBmNav)

    @staticmethod
    def mf_bm_info():
        """
        Query ChinaMutualFundBenchMark.
        Returns
        -------

        """
        return sess_wind.query(MfBmInfo)

    @staticmethod
    def index_wind_ind():
        """
        Query AIndexWindIndustriesEOD.
        Returns
        -------

        """
        return sess_wind.query(IndexWindInd)

    @staticmethod
    def index_ashare():
        """
        Query AIndexEODPrices.
        Returns
        -------

        """
        return sess_wind.query(AIndexPrices)

    @staticmethod
    def index_bond():
        """
        Query CBIndexEODPrices.
        Returns
        -------

        """
        return sess_wind.query(BondIndex)

    @staticmethod
    def index_bond_info():
        """
        Query CBindexDescription.
        Returns
        -------

        """
        return sess_wind.query(CBIndexDesc)

    @staticmethod
    def index_ashare_info():
        """
        Query AIndexDescription.
        Returns
        -------

        """
        return sess_wind.query(AIndexDesc)

    @staticmethod
    def index_futures():
        """
        Query CIndexFuturesEODPrices.
        Returns
        -------

        """
        return sess_wind.query(CfePrices)


class FundManager:

    def __init__(self, manager_name):

        if not isinstance(manager_name, str):
            raise TypeError("manager name must be string!")
        self.manager_name = manager_name
        self.codes = self.manager_products()['F_INFO_WINDCODE'].unique().tolist()

    def manager_products(self, current=True, cn_name=False):
        """
        Query products information of the manager.

        Parameters
        ---------
        current : bool
                if current is True, we'll query the funds under the manager
                currently; if current is False, we'll query all the funds including
                current and historical funds under the manager.
        cn_name : bool

        Returns
        -------
        mg_products : pd.DataFrame

        Examples
        -------
        >>> manager_name = "乐无穹"
        >>> manager_name = '刘彦春'
        >>> fm = FundManager(manager_name)
        >>> mg_products = fm.manager_products()
        >>> mg_products = fm.manager_products(current=False)
        >>> mg_products = fm.manager_products(cn_name=True)
        """
        products_query: Query = sess_wind.query(MfManager, MfInfo.F_INFO_NAME).\
            join(MfInfo, MfManager.F_INFO_WINDCODE == MfInfo.F_INFO_WINDCODE).\
            filter(MfManager.F_INFO_FUNDMANAGER == self.manager_name,
                   MfManager.F_INFO_MANAGER_STARTDATE.isnot(None))

        products_query = products_query.options(Load(MfManager).load_only('F_INFO_FUNDMANAGER', 'F_INFO_FUNDMANAGER_ID',
                                                                          'F_INFO_WINDCODE', 'F_INFO_MANAGER_STARTDATE',
                                                                          'F_INFO_MANAGER_LEAVEDATE'),
                                                Load(MfInfo).load_only('F_INFO_NAME'))

        if current:
            products_query = products_query.filter(MfManager.F_INFO_MANAGER_LEAVEDATE.is_(None))

        products = DataBase.query_to_df(products_query, cn_name)

        return products

    def manager_scale(self):
        """
        Cal management scale of the manager..

        Returns
        ------
        mg_scale : float
            Management scale of the manager. the measurement unit is '亿元'.

        Examples
        -------
        >>> name = '刘彦春'
        >>> name = '乐无穹'
        >>> fm = FundManager(name)
        >>> MF.scale_latest(fm.codes)
        >>> mg_scale = fm.manager_scale()
        """
        mg_pro_scale = MF.scale_latest(self.codes)
        mg_scale = sum(mg_pro_scale['FUND_SCALE'])

        return mg_scale

    def products_nav(self, fund_name=True):
        """
        Get manager products nav. Products nav should fall in
        range between management start date and end date.

        Parameters
        ---------
        fund_name : bool

        Returns
        ------
        mg_pro_nav_sub : pd.DataFrame
                      manager products nav that fall in range between management start date and end date.

        Examples
        -------
        >>> name = "乐无穹"
        >>> name = '刘彦春'
        >>> fm = FundManager(name)
        >>> fund_name = True
        >>> mg_pro_nav_sub = fm.products_nav()
        >>> mg_pro_nav_sub = fm.products_nav(fund_name=False)
        """
        mg_products = self.manager_products()[['F_INFO_WINDCODE', 'F_INFO_MANAGER_STARTDATE', 'F_INFO_MANAGER_LEAVEDATE']]
        products_dict = mg_products.set_index('F_INFO_WINDCODE').to_dict(orient='index')

        nav_list = []
        for i in self.codes:
            start, end = products_dict[i]['F_INFO_MANAGER_STARTDATE'], products_dict[i]['F_INFO_MANAGER_LEAVEDATE']
            nav_list.append(MF.nav(i, start=start, end=end, fund_name=fund_name))
        nav = pd.concat(nav_list).reset_index(drop=True)

        return nav

    def manager_performance(self, periods=['SI', '1Y', '2Y', '3Y', '5Y']):
        """
        Analysis of mutual fund manager products. By default, we calculate 'SI'、 '1Y'、 '2Y'、 '3Y'、 '5Y'
        performance and integer year like '2019'、'2018' and so on.

        Parameters
        ---------
        periods : list
                time range list.This function will analyze products in different time range.

        Returns
        -------
        results_df : pd.DataFrame
                   indicators of all the products of the manager in different time range.

        Examples
        --------
        >>> name = "乐无穹"
        >>> name = "刘彦春"
        >>> fm = FundManager(name)
        >>> mg_pro_nav = fm.products_nav(fund_name=False)
        >>> mg_fund_bm_info = MF.bm_index_info(fm.codes, cn_name=False)
        >>> mg_bm_nav = MF.bm_index_nav(fm.codes, fund_name=False)
        >>> mg_ana = fm.manager_performance()
        >>> mg_ana = fm.manager_performance(periods = ['SI', '1Y', '2Y'])
        """
        # mg_pro_codes = self.manager_products().F_INFO_WINDCODE.unique().tolist()

        mg_pro_nav = self.products_nav(fund_name=False)
        mg_pro_nav['DATETIME'] = DateUtil.parse_dates(mg_pro_nav['DATETIME'])
        mg_nav_instance = Nav(mg_pro_nav, freq='weekly')

        mg_fund_bm_info = MF.bm_index_info(self.codes, cn_name=False)
        mg_fund_bm_map = dict(zip(mg_fund_bm_info['F_INFO_WINDCODE'], mg_fund_bm_info['BENCHMARK_WINDCODE']))

        mg_bm_nav = MF.bm_index_nav(self.codes, fund_name=False)
        mg_bm_nav['DATETIME'] = DateUtil.parse_dates(mg_bm_nav['DATETIME'])
        mg_bm_rbs = Nav(mg_bm_nav, freq='daily').ret()

        mg_info_query = sess_wind.query(MfManager).filter_by(F_INFO_FUNDMANAGER=self.manager_name)
        mg_info = DataBase.query_to_df(mg_info_query)
        manage_start_date = parse(min([x for x in mg_info['F_INFO_MANAGER_STARTDATE'] if x is not None])).date()
        yearly_date = pd.date_range(manage_start_date, DateUtil.add_date(datetime.datetime.today(), '+1y'), freq='Y')
        year = list(map(lambda date: str(date.year), yearly_date))
        periods += year

        results_dict = mg_nav_instance.cal_metrics(rbs=mg_bm_rbs, bm_codes=mg_fund_bm_map, periods=periods)
        results_df = PdUtil.select(pd.DataFrame(results_dict), MetricsMap.metrics_cn_name)
        results_df.replace({'产品名称': dict(zip(mg_fund_bm_info['F_INFO_WINDCODE'], mg_fund_bm_info['F_INFO_NAME']))}, inplace=True)
        results_df.replace({'基准': dict(zip(mg_fund_bm_info['BENCHMARK_WINDCODE'], mg_fund_bm_info['F_INFO_BENCHMARK']))}, inplace=True)

        return results_df


class MF:

    @staticmethod
    def fund_info(codes, *args, cn_name=False, **kwargs):
        """
       Query fund information.
       Attention: You columns select must accord column name after you set cn_name True or Flase.
       For example, when you set cn_name True, you select columns must in chinese column name.

        Parameters
        ---------
        codes : Union[list, str]
              mutual fund codes
        cn_name : bool

        Returns
        -------
        fund_info : pd.DataFrame

        Examples
        --------
        >>> codes = "159934.SZ"
        >>> codes = ["159934.SZ", "159937.SZ"]
        >>> cols = ["F_INFO_WINDCODE", "F_INFO_NAME", "F_INFO_CORP_FUNDMANAGEMENTCOMP",
                 "F_INFO_SETUPDATE", "F_INFO_MATURITYDATE"]
        >>> fund_info = MF.fund_info(codes, *cols)
        >>> fund_info = MF.fund_info(codes, "F_INFO_WINDCODE", 名称="F_INFO_NAME")
        >>> fund_info = MF.fund_info(codes, "F_INFO_WINDCODE", "F_INFO_NAME")
        >>> fund_info = MF.fund_info(codes,  'Wind代码', '基金品种ID', cn_name=True)
        >>> fund_info = MF.fund_info(codes, cn_name=True)
        >>> fund_info = MF.fund_info(codes)
        >>> fund_info = MF.fund_info(codes, *['Wind代码', '基金品种ID'], cn_name=True)
        """

        if isinstance(codes, str):
            fund_info_query = sess_wind.query(MfInfo).filter(MfInfo.F_INFO_WINDCODE == codes)
        else:
            fund_info_query = sess_wind.query(MfInfo).filter(MfInfo.F_INFO_WINDCODE.in_(codes))

        fund_info = DataBase.query_to_df(fund_info_query, cn_name)

        if args or kwargs:
            fund_info = PdUtil.select(fund_info, *args, **kwargs)

        return fund_info

    @staticmethod
    def various_navs(codes, start=None, end=None, cn_name=False):
        """
        Query various navs mutual fund from wind database.

        Parameters
        ----------
        codes : Union[string, list]
        start : Union[string, date]
        end : Union[string, date]
        cn_name : bool
                Set chinese column name for data frame when cn_name is True.

        Returns
        -------
        fund_nav : pd.DataFrame
                  mutual fund nav including unit nav, accumulated nav and adjusted nav..

        Examples
        --------
        >>> codes = "159934.SZ"
        >>> codes = ["159934.SZ", "159937.SZ"]
        >>> fund_nav = MF.various_navs(codes)
        >>> fund_nav = MF.various_navs(codes, cn_name=True)
        >>> fund_nav = MF.various_navs(codes, cn_name=True, start='20190101', end='20190801')
        >>> fund_nav = MF.various_navs(codes, cn_name=True, start='2019-01-01', end='20190901')
        """
        nav_columns = MfNav.PRICE_DATE, MfNav.F_INFO_WINDCODE, MfNav.F_NAV_UNIT, MfNav.F_NAV_ACCUMULATED, \
                      MfNav.F_NAV_ADJUSTED
        if isinstance(codes, str):
            nav_query = sess_wind.query(*nav_columns).filter(MfNav.F_INFO_WINDCODE == codes,
                                                            SaUtil.filter_date(MfNav.PRICE_DATE, start, end, 'YYYYMMDD'))
        else:
            nav_query = sess_wind.query(*nav_columns).filter(MfNav.F_INFO_WINDCODE.in_(codes),
                                                            SaUtil.filter_date(MfNav.PRICE_DATE, start, end, 'YYYYMMDD'))
        fund_nav = DataBase.query_to_df(nav_query, cn_name)
        return fund_nav

    @staticmethod
    def nav(codes, start=None, end=None, fund_name=False):
        """
        Query fund nav.

        Parameters
        ---------
        codes : Union[str, list]
        start : Union[string, date]
        end : Union[string, date]
        fund_name : bool
            Use name of fund as code when fund_name is True.

        Returns
        ------
        fund_nav : pd.DataFrame

        Examples
        -------
        >>> codes = "159934.SZ"
        >>> codes = ["159934.SZ", "159937.SZ"]
        >>> fund_nav = MF.nav(codes)
        >>> fund_nav = MF.nav(codes, fund_name=True)
        >>> fund_nav = MF.nav(codes, fund_name=True, start='20190101', end='20190801')
        """
        nav_columns = MfNav.PRICE_DATE, MfNav.F_INFO_WINDCODE, MfNav.F_NAV_ADJUSTED

        if isinstance(codes, str):
            nav_query = sess_wind.query(*nav_columns).filter(MfNav.F_INFO_WINDCODE == codes,
                                                             SaUtil.filter_date(MfNav.PRICE_DATE, start, end, 'YYYYMMDD'))
        else:
            nav_query = sess_wind.query(*nav_columns).filter(MfNav.F_INFO_WINDCODE.in_(codes),
                                                             SaUtil.filter_date(MfNav.PRICE_DATE, start, end, 'YYYYMMDD'))
        fund_nav = DataBase.query_to_df(nav_query).rename(columns={'PRICE_DATE': 'DATETIME', 'F_INFO_WINDCODE': 'code',
                                                           'F_NAV_ADJUSTED': 'value'})

        if fund_name:
            fund_info = MF.fund_info(codes, 'F_INFO_WINDCODE', 'F_INFO_NAME')
            name_dict = dict(zip(fund_info['F_INFO_WINDCODE'], fund_info['F_INFO_NAME']))
            fund_nav.replace({'code': name_dict}, inplace=True)

        return fund_nav.sort_values(by=['code', 'DATETIME']).reset_index(drop=True)

    @staticmethod
    def bm_index_code(code):
        """
        Generate fund benchmark index wind code.

        Parameters
        ----------
        code : str

        Returns
        -------
        bm_code : str

        Examples
        --------
        >>> code = "159934.SZ"
        >>> MF.bm_code(code)
        """
        bm_code = code[0:6] + 'BI.WI'
        return bm_code

    @staticmethod
    def bm_index_info(codes, cn_name=False):
        """
        Query benchmark index information.

        Parameters
        ----------
        codes : Union[str,list]
        cn_name : bool

        Returns
        -------
        bm_index_info : pd.DataFrame

        Examples
        -------
        >>> codes = ["159934.SZ", "159937.SZ"]
        >>> codes = ["007153.OF", "007154.OF"]
        >>> bm_index_info = MF.bm_index_info(codes)
        >>> bm_index_info = MF.bm_index_info(codes, cn_name=True)
        """
        # TODO 2019/9/17 13:26 wgs: ASHAREDESCRIPTION can be used here.
        bm_index_info = MF.fund_info(codes, cn_name=False)[['F_INFO_WINDCODE', 'F_INFO_NAME', 'F_INFO_BENCHMARK']]
        bm_index_info['BENCHMARK_WINDCODE'] = bm_index_info.F_INFO_WINDCODE.apply(lambda x: MF.bm_index_code(x))

        if cn_name:
            bm_index_info.rename(columns={'F_INFO_WINDCODE': 'Wind代码', 'F_INFO_NAME': '基金简称',
                                'F_INFO_BENCHMARK': '业绩比较基准', 'BENCHMARK_WINDCODE': '基准Wind代码'}, inplace=True)
        return bm_index_info

    @staticmethod
    def bm_component_info(codes, cn_name=False):
        """
        Query mutual fund benchmark component information.

        Parameters
        ---------
        codes : Union[str, list]
        cn_name : bool

        Returns
        -------
        mf_bm_info : pd.DataFrame

        Examples
        --------
        >>> codes = "159934.SZ"
        >>> codes = ["159934.SZ", "159937.SZ"]
        >>> codes = ["007153.OF", "007154.OF"]
        >>> bm_comp = MF.bm_component_info(codes)
        >>> bm_comp = MF.bm_component_info(codes, cn_name=True)
        """
        if isinstance(codes, str):
            bm_component_query = sess_wind.query(BmComponent).filter(BmComponent.S_INFO_WINDCODE == codes)
        else:
            bm_component_query = sess_wind.query(BmComponent).filter(BmComponent.S_INFO_WINDCODE.in_(codes))

        bm_component_info = DataBase.query_to_df(bm_component_query, cn_name)
        return bm_component_info

    @staticmethod
    def bm_index_nav(codes, start=None, end=None, fund_name=False):
        """
        Query mutual fund bm index nav.

        Parameters
        ----------
        codes: Union[str, list]
              mutual fund codes.
        start : Union[str, date]
        end: : Union[str, date]
        fund_name :  bool

        Returns
        -------
        out : pd.DataFrame
            mutual fund benchmark nav with three columns: 'DATETIME', 'code', 'value'.

        Examples
        --------
        >>> codes = "159934.SZ"
        >>> codes = ["159934.SZ", "159937.SZ"]
        >>> codes = ["007153.OF", "007154.OF"]
        >>> bm_nav = MF.bm_index_nav(codes)
        >>> bm_nav = MF.bm_index_nav(codes, fund_name=True)
        >>> bm_nav = MF.bm_index_nav(codes, fund_name=True, start='20190101', end='20190801')
        """
        if isinstance(codes, str):
            codes = [codes]
        bm_codes = [MF.bm_index_code(i) for i in codes]

        bm_nav_query = sess_wind.query(MfBmNav).filter(MfBmNav.S_INFO_WINDCODE.in_(bm_codes),
                                                       SaUtil.filter_date(MfBmNav.TRADE_DT, start, end, 'YYYYMMDD'))
        bm_nav = DataBase.query_to_df(bm_nav_query)
        bm_nav = PdUtil.select(bm_nav, DATETIME='TRADE_DT', code='S_INFO_WINDCODE', value='S_DQ_CLOSE')

        if fund_name:
            bm_index_info = MF.bm_index_info(codes)
            name_dict = dict(zip(bm_index_info['BENCHMARK_WINDCODE'], bm_index_info['F_INFO_BENCHMARK']))
            bm_nav.replace({'code': name_dict}, inplace=True)

        bm_nav = bm_nav.sort_values(by=['code', 'DATETIME']).reset_index(drop=True)
        return bm_nav

    @staticmethod
    def shares(codes, start=None, end=None, cn_name=False):
        """
        Query mf shares.

        Parameters
        ----------
        codes : Union[str, list]
        start : Union[str, date]
        end : Union[str, date]
        cn_name : bool

        Returns
        -------
        shares : pd.DataFrame

        Examples
        ------
        >>> codes = "159934.SZ"
        >>> shares = MF.shares(codes)
        >>> shares = MF.shares(codes, cn_name=True)
        """
        query_cols = MfShare.F_INFO_WINDCODE, MfShare.CHANGE_DATE, MfShare.FUNDSHARE
        if isinstance(codes, str):
            shares_query = sess_wind.query(*query_cols).filter(MfShare.F_INFO_WINDCODE == codes)
        else:
            shares_query = sess_wind.query(*query_cols).filter(MfShare.F_INFO_WINDCODE.in_(codes))

        shares_query = shares_query.filter(SaUtil.filter_date(MfShare.CHANGE_DATE, start, end, 'YYYYMMDD'))

        shares = DataBase.query_to_df(shares_query, cn_name)
        return shares

    @staticmethod
    def shares_latest(codes, last_quarter=False, cn_name=False): # 672 error
        """
        Query latest mutual fund shares.

        Parameters
        ----------
        codes : Union[str, list]
        last_quarter : bool
        cn_name : bool

        Returns
        -------
        out : pd.DataFrame
            latest mutual fund shares.

        Examples
        --------
        >>> codes = "159934.SZ"
        >>> codes = ["159934.SZ", "159937.SZ"]
        >>> codes = ["512650.SH", "515803.SH", "007153.OF", "007154.OF"]
        >>> fund_share = MF.shares_latest(codes)
        >>> fund_share = MF.shares_latest(codes, cn_name=True, last_quarter=True)
        """
        start = DateUtil.add_date(datetime.date.today(), '-1q')
        if last_quarter:
            end = DateUtil.add_date(datetime.date.today(), '_fq-1d')
        else:
            # end = DateUtil.add_date(datetime.date.today(), '-1d')
            end = datetime.date.today()

        shares = MF.shares(codes, start, end)
        shares = PdUtil.group_max_rows(shares, 'F_INFO_WINDCODE', 'CHANGE_DATE').reset_index(drop=True)

        if cn_name:
            shares.rename(columns=Dict.query_dict('CHINAMUTUALFUNDSHARE'), inplace=True)

        return shares

    @staticmethod
    def scale(codes, start=None, end=None, cn_name=False):
        """
        Calculate fund scale.

        Parameters
        ----------
        codes : Union[str, list]
        start : Uinon[str, date]
        end : Union[str, date]
        cn_name : bool

        Returns
        -------
        scale : pd.DataFrame

        Examples
        -------
        >>> codes = ['169106.SZ', '002803.OF']
        >>> scale = MF.scale(codes, '20190101', '20190630')
        >>> scale = MF.scale(codes, '2019-01-01', '2019-06-30', cn_name=True)
        """
        shares = MF.shares(codes, start, end).rename(columns={'CHANGE_DATE': 'PRICE_DATE'})
        nav = MF.various_navs(codes, start, end)[['F_INFO_WINDCODE', 'PRICE_DATE', 'F_NAV_UNIT']]

        shares_nav = shares.merge(nav, how='left', on=['F_INFO_WINDCODE', 'PRICE_DATE'])
        shares_nav['FUND_SCALE'] = shares_nav['FUNDSHARE'] * shares_nav['F_NAV_UNIT'] / 10000 # type: pd.DataFrame

        scale = shares_nav[['F_INFO_WINDCODE', 'PRICE_DATE', 'FUND_SCALE']].\
            sort_values(by=['F_INFO_WINDCODE', 'PRICE_DATE']).dropna().reset_index(drop=True)
        if cn_name:
            scale.rename(columns={'F_INFO_WINDCODE': 'Wind代码', 'PRICE_DATE': '截止日期',
                                  'FUND_SCALE': '基金规模'}, inplace=True)
        return scale

    @staticmethod
    def scale_latest(codes=None, cn_name=False):
        """
        Calaculate the latest fund scale.

        Parameters
        ----------
        codes : Union[str, list]
        cn_name : bool

        Returns
        -------
        scale : pd.DataFrame

        Examples
        --------
        >>> codes = "159934.SZ"
        >>> codes = ["512650.SH", "515803.SH", "007153.OF", "007154.OF"]
        >>> fund_scale = MF.scale_latest(codes)
        >>> fund_scale = MF.scale_latest(codes, cn_name=True)
        """
        start = DateUtil.add_date(datetime.date.today(), '-1q')
        end = datetime.date.today()

        scale = MF.scale(codes, start=start, end=end)
        scale = PdUtil.group_max_rows(scale, 'F_INFO_WINDCODE', 'PRICE_DATE').reset_index(drop=True)

        if cn_name:
            scale.rename(columns={'F_INFO_WINDCODE': 'Wind代码', 'PRICE_DATE': '截止日期',
                                  'FUND_SCALE': '基金规模'}, inplace=True)
        return scale


class WindIndex:
    """
    Wind Index class provide some function to query index info and nav.

    Examples
    -------
    >>> index_code = '000300.SH'
    >>> index_test = WindIndex(index_code)
    >>> index_test.futures_code

    """

    def __init__(self, index_code):
        self.code = index_code

    @staticmethod
    def futures_code(code):
        """
        Return stock index futures code.
        """
        futures_dict = {'000300.SH': 'IF.CFE',
                        '000905.SH': 'IC.CFE',
                        '000016.SH': 'IH.CFE'
                        }

        return futures_dict.get(code, None)

    def info(self, info_table='AIndexDesc', cn_name=False):
        """
        Get wind index information.

        Returns
        -------
        out : dict
            dict that contains wind code and name of the index.

        Examples
        -------
        >>> code = "000300.SH"
        >>> code = "007153BI.WI"
        >>> code = "IC.CFE" # error
        >>> wi = WindIndex(code)
        >>> index_info = wi.info()
        >>> index_info = wi.info(cn_name=True) # NO DICT AINDEXDESCRIPTION
        """
        info_table_map = {
            'AIndex': AIndexDesc,
            'BondIndex': CBIndexDesc
        }

        if info_table not in info_table_map.keys():
            raise ValueError("info_table must be in {'AIndexDesc', 'BondIndexDesc'}.")

        info_query = sess_wind.query(info_table_map.get(info_table)).filter_by(S_INFO_WINDCODE=self.code)

        index_info = DataBase.query_to_df(info_query, cn_name=cn_name)

        return index_info

    def market_data(self, *args, start=None, end=None, **kwargs):
        """
        Query market data such as close price、volume、amount and so on of index.

        Parameters
        ----------
        kwargs :
               it can be different columns according to requirements. Notice that columns selected
               will be ignored if they are non-existent in the database table .
               e.g. 007153BI.WI has no volume.

        Returns
        -------
        index_md : pd.DataFrame()
                 data frame with at least two columns. ['DATETIME', 'code']

        Examples
        --------
        >>> code = "000300.SH"
        >>> code = "007153BI.WI"
        >>> code = "IC.CFE"
        >>> wi = WindIndex(code)
        >>> index_md = wi.market_data(value = 'S_DQ_CLOSE')
        >>> index_md = wi.market_data('S_DQ_CLOSE')
        >>> index_md = wi.market_data(value='S_DQ_CLOSE', volume='S_DQ_VOLUME')
        >>> index_md = wi.market_data(value='S_DQ_CLOSE', volume='S_DQ_VOLUME', start='20190101', end='20190901')
        """
        columns, column_dict = columns_from_args(*args, **kwargs)

        md_table_map = {
            '.WI': IndexWindInd,
            'BI.WI': MfBmNav,
            '.CFE': CfePrices,
            'CB': BondIndex,
            '.HI': HKIndexPrices
        }

        fix_match = re.match(".*?(?P<fix>(.WI)|(BI.WI)|(.CFE)|(.HI)|(CB)|(.SH)|(.SZ)|(.CSI))", self.code)
        md_table = md_table_map.get(fix_match['fix'], AIndexPrices)

        query = sess_wind.query(md_table).filter_by(S_INFO_WINDCODE=self.code).options(load_only(*columns))
        query = query.filter(SaUtil.filter_date(md_table.TRADE_DT, start, end, 'YYYYMMDD'))

        md_df = DataBase.query_to_df(query)
        column_dict.update({'TRADE_DT': 'DATETIME', 'S_INFO_WINDCODE': 'code'})
        md_df.rename(columns=column_dict, inplace=True)
        return md_df.sort_values(by='DATETIME')

    def close_prices(self, start=None, end=None):
        """
        Query index close price.

        Parameters
        ---------
        start : str
        end : str

        Returns
        ------
        index_close : pd.DataFrame
                    close price of index.

        Examples
        -------
        >>> code = "000300.SH"
        >>> code = "007153BI.WI"
        >>> start = "20190101"
        >>> end = "20190826"
        >>> wi = WindIndex(code)
        >>> index_close = wi.close_prices(start, end)
        """
        index_close = self.market_data(value='S_DQ_CLOSE', start=start, end=end)

        return index_close

    def trading_volume(self, start=None, end=None):
        """
        Query index trading volume.

        Parameters
        ---------
        start : str
        end : str

        Returns
        ------
        index_volume : pd.DataFrame
                    trading volume of index.

        Examples
        --------
        >>> code = "000300.SH"
        >>> start = "20190101"
        >>> end = "20190801"
        >>> wi = WindIndex(code)
        >>> index_vol= wi.trading_volume(start, end)
        """
        index_volume = self.market_data(volume='S_DQ_VOLUME', start=start, end=end)

        return index_volume

    def trading_amount(self, start=None, end=None):
        """
        Query index trading amount.

        Parameters
        ---------
        start : str
        end : str

        Returns
        -------
        index_amount : pd.DataFrame
                    trading amount of index.

        Examples
        --------
        >>> code = "000300.SH"
        >>> start = "20190101"
        >>> end = "20190801"
        >>> wi = WindIndex(code)
        >>> index_amount= wi.trading_amount(start, end)
        """
        index_amount = self.market_data(amount='S_DQ_AMOUNT', start=start, end=end)

        return index_amount
