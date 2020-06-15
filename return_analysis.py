# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:27:13 2020

@author: Gao haocheng
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_columns',24)
pd.set_option('display.max_rows',100)

ENV_PATH = 'C:\\Users\\apply\\OneDrive - National University of Singapore\\Stock_CSV\\'
#ENV_PATH = 'D:\\OneDrive - National University of Singapore\\Stock_CSV\\'

exchange_map = {'NYSE':1, 'NYSE MKT':2, 'NASDAQ':3, 'NYSE Arca':4}


def get_forward_return( signal_data, start, end, signal_lag, exclude_penny_stock=False , value_weighted=False, exchanges='all', holding_period = [1]):
    '''
    Compute fwd return

    Parameters
    ----------
    signal_data : DataFrame from read_csv. 
        Has following columns:['DATE','PERMNO','SIGNAL']
        
    start : DataTime64
        Start date of portfolio
        
    end : DateTime64
        End date of portfolio
        
    signal_lag: int
        The lag between formation period and holding period
        
    exclude_penny_stock: bool
        Whether need to remove stocks which price<5 in the month end
    
    value_weighted: bool
        Equal weighted or market value weighted
        
    exchanges: list or 'all'
        Specify the exchanges
        
    holding_period : list, optional
        Each number in this list represent a fwd period which will calculate return. The default is [1].
    

    Returns
    -------
    MultiIndex DataFrame:
                    ------------------------------------------
                                      1                    2                 SIGNAL
    DATE       PERMNO                                                      
    2017-01-31 10000.0             0.039825            -0.028349            18.974852
               20000.0            -0.078815            -0.312000            11.705812
               30000.0            -0.106814            -0.076243            17.797900
                                    ...                  ...                   ...
                   ------------------------------------------
    '''

    data = signal_data.copy()
    data = data[['DATE','PERMNO','SIGNAL']]
    #data['DATE'] = pd.to_datetime(data['DATE'],format = '%m/%d/%Y')
    #data['DATE'] = pd.to_datetime(data['DATE'],format = '%Y%m%d')
    data['DATE'] = pd.to_datetime(data['DATE'],format = '%Y-%m-%d')
    data['DATE'] = data['DATE'] - pd.offsets.MonthBegin(1) + pd.offsets.MonthEnd(1)
    data = data[(data['DATE']>=start) & (data['DATE']<=end)]
    # move all date to the end of that month
    
    original_length = len(data)
    print('\n%d observations in original dataset\n'%original_length)
    # print length of data    
    
    month_ret = pd.read_csv(ENV_PATH + 'msf.csv')
    month_ret = month_ret[['DATE','PERMNO','RET','PRC','HEXCD','SHROUT']]
    month_ret['RET'] = pd.to_numeric(month_ret['RET'], errors='coerce')
    month_ret['DATE'] = pd.to_datetime(month_ret['DATE'], format = '%Y%m%d')
    month_ret['DATE'] = month_ret['DATE'] - pd.offsets.MonthBegin() + pd.offsets.MonthEnd(1)
    # import monthly return
    
    max_hd = max(holding_period)
    col = ['DATE','PERMNO'] + [i for i in range(1,max_hd+1)] + ['SIGNAL']
    
    if exclude_penny_stock:
        data = data.merge(month_ret, on = ['DATE','PERMNO'], how = 'left')
        data = data[data['PRC'].abs()>5]
        data = data[['DATE','PERMNO','SIGNAL']]
        drop_percentage = (1 - len(data)/original_length)*100
        print('%0.1f%% observations has been droped after removed penny stocks\n'%drop_percentage)
        
    if exchanges!='all':
        data = data.merge(month_ret, on = ['DATE','PERMNO'], how = 'left')
        data = data[data['HEXCD'].isin(exchanges)]
        data = data[['DATE','PERMNO','SIGNAL']]
        drop_percentage = (1 - len(data)/original_length)*100
        print('%0.1f%% observations has been droped after specified exchanges\n'%drop_percentage)
        
    if value_weighted:
        month_ret['mkt_value'] = month_ret['PRC'].abs().mul(month_ret['SHROUT'])*1000
        data = data.merge(month_ret, on = ['DATE','PERMNO'], how = 'left')
        data = data[['DATE','PERMNO','SIGNAL','mkt_value']]
        col = col + ['mkt_value']
    # print length of data    
    month_ret = month_ret[['DATE','PERMNO','RET']]

    for i in range(1,max_hd+1):
        next_i_month_ret = month_ret.copy()
        next_i_month_ret['DATE'] = next_i_month_ret['DATE'] - pd.offsets.MonthEnd(i+signal_lag)
        next_i_month_ret.rename(columns={'RET':i},inplace=True)
        
        data = data.merge(next_i_month_ret, on = ['PERMNO','DATE'], how = 'left')
        # merge next x months return to dataframe
        

    data = data[col]
    
    data.set_index(['DATE','PERMNO'],inplace=True)
    
    data.sort_index(inplace=True)
    
    return data
        
    
def get_quantile( data, quantile = 10, groupby = None, quantize_by_group = False):
    '''
    Get the quantile in each month

    Parameters
    ----------
    data : data from 'get_forward_return'

    quantile : int or a list, optional
        The number of groups you want to divide the stocks into.
        For a list, it means the quantile of signal data. Like [0, 0.1, 0.5, 0.7, 1]. 
        The default is 10, which is decile.
        
    groupby : a map, optional
        A map which key is the name of stocks, value is the group to which stock belongs. The default is None.
        
    quantize_by_group : bool, optional
        Whether you want to divide stocks in each group. The default is False.

    Returns
    -------
    MultiIndex DataFrame:
                            ------------------------------------------
                                      1                    2                SIGNAL     group    factor_quantile
    DATE       PERMNO                                                             
    2015-02-28 13766.0             0.051952            -0.025555          16.017964      2          4
               13748.0            -0.028338            -0.100756          19.862959      2          5
               54704.0             0.039352            -0.051697           7.974977      2          2
               21936.0             0.013695            -0.011364           8.468198      2          2
               13743.0            -0.055304             0.005439          15.470253      2          3
                            ------------------------------------------
    '''
    data_copy = data.copy()
    
    if groupby is not None:
        diff = set(groupby.keys())-set(data_copy.index.get_level_values('PERMNO'))
        if diff:
            raise KeyError(
                    "Assets {} not in group mapping".format(list(diff)))
        groupby = pd.Series(groupby)
        groupby = groupby.reindex(data_copy.index.get_level_values('PERMNO').to_series())
        data_copy['group'] = groupby.values
    # exam if all stocks has a group
        
    def _quantize_factor(_x, _quantile):
        
        sort_x = _x.sort_values().to_frame(name = 'signal')
        num_in_decile = len(sort_x) // _quantile
        top_threshold = sort_x.iloc[-num_in_decile,0]
        bottom_threshold = sort_x.iloc[num_in_decile-1,0]

        sort_x.loc[sort_x['signal']>=top_threshold,'quantile'] = _quantile
        sort_x.loc[sort_x['signal']<=bottom_threshold,'quantile'] = 1
        
        try:
            remain_value = sort_x.loc[sort_x['quantile'].isna(),'signal']
            sort_x.loc[sort_x['quantile'].isna(), 'quantile'] = pd.qcut(remain_value.rank(method='first'), _quantile-2, labels = False) +2
            # for some same signal in the junction, move them to other quantile    
            sort_x = sort_x.reindex(_x.index)
            return sort_x['quantile']
        
        except ValueError:
            print('too small stocks in %s to generate portfolio\n'%(_x.index.get_level_values('DATE')[0].strftime('%d %b, %Y')))
            return pd.DataFrame(index=_x.index)
    # quantize factor by quantile
    
    grouper = [data_copy.index.get_level_values('DATE')]
    if quantize_by_group:
        grouper.append('group')
    # add a criterion when user want see performance in different groups
    
    quantile_data = data_copy.groupby(grouper)['SIGNAL'].apply(_quantize_factor,_quantile=quantile)
    # generate quantile
    
    data_copy['factor_quantile'] = quantile_data
    
    print('%.4f%% has been drop when quantilize stocks\n'%(1-len(data_copy.dropna(subset=['factor_quantile']))/len(data_copy)))
    
    data_copy = data_copy.dropna(subset=['factor_quantile'])
    
    return data_copy


def get_weight( stock_data, weight_scheme = 'equal_weighted', quantile = 'all', hurdle = 0, calculate_by_group = False):
    '''
    Calculate the weight

    Parameters
    ----------
    stock_data : DataFrame
        result of 'get_quantile'
        
    weight_scheme : str ['equal_weighted','value_weighted','factor_weighted']
        Equal weighting, market value weighting or factor value weighted

    quantile : 'all' or int, optional
        The quantile you concern about. 
        int is the quantile number.
        
    hurdle : int, optional
        The number of stocks in each cross section. 
        We calculate the return only when the number of observations in that cross section higher than this hurdle.
        The default is 0.
        
    calculate_by_group : bool, optional
        Whether need to calculate weight in different groups.
        If True, the sum of abs(weight) of each group in each cross section is equals to 1 
        The default is False
        
    Returns
    -------
    A MiltiIndex DataFrame

    '''
    data = stock_data.copy()
    
    if quantile != 'all':
        data = data[data['factor_quantile']==quantile]
    # select quantile data
    
    grouper = [data.index.get_level_values('DATE')]
    
    if calculate_by_group:
        grouper.append('group')
        
    grouper.append('factor_quantile')
    
    def _calculate_weights( _data, _weight_scheme):
        
        if len(_data)<hurdle:
            return pd.DataFrame([np.nan]*len(_data), index = _data.index, columns=['weight'])
        # if there is not enough data, skip the section
            
        if _weight_scheme == 'equal_weighted':
            
            weight = pd.Series([1/len(_data)] * len(_data), index = _data.index)
            
        elif _weight_scheme == 'value_weighted':
            
            weight = _data['mkt_value']/_data['mkt_value'].sum()

        elif _weight_scheme == 'factor_weighted':
            
            weight = _data['SIGNAL']/_data['SIGNAL'].abs().sum()
            
        return (weight / weight.abs().sum()).to_frame('weights')
        # make the sum of abs(weight) equals to 1
        
    weights = data.groupby(grouper)[['mkt_value']] if weight_scheme == 'value_weighted' else data.groupby(grouper)[['SIGNAL']]
    
    weights = weights.apply(_calculate_weights, weight_scheme)
    
    data['weights'] = weights
    
    print('%0.1f%% has been drop when calculate weight\n'%(1-len(data.dropna(subset=['weights']))/len(data)))
    
    return data.dropna(subset=['weights'])    



def rolling_return_helper(sub_portfolio_return, start, holding_period):
    '''
    Use this function to calculate the rolling return of a portfolio when holding period and factor calculation period is different.

    Parameters
    ----------
    sub_portfolio_return : DataFrame
        Return of the sub-portfolio from the day it was created to the day it was sold.
                                ------------------------------------------
                                    1         2         3         4         5         6   
                     DATE                                                                     
                    2009-02-28  0.040940  0.071787  0.015230 -0.001500  0.020210  0.004570   
                    2009-03-31  0.092983  0.013958 -0.011742  0.026696  0.015977 -0.002707   
                    2009-04-30  0.005977 -0.016665  0.033173  0.023590  0.013031 -0.030905   
                    2009-05-31 -0.015705  0.019665  0.014298  0.015412 -0.025815  0.007290   
                    2009-06-30  0.004832 -0.008482  0.000318 -0.031033 -0.003411  0.015833 
                                ------------------------------------------
        Index is the start time of a portfolio. 
        Each column is the return of that sub-portfolio in the next x month .
        Must has more than max(holding_period) columns
        
    start : datetime64
        Start date of calculating rolling return.
        
    holding_period : list
        Expected holding period.

    Returns
    -------
    portfolio_return: DataFrame
                                ------------------------------------------
                                 holding period       1             6 
                                 DATE              
                                2010-01-31       0.0086768       0.0090952
                                2010-02-28      -0.0021307       0.0138947
                                2010-03-31       0.0124681       0.0246391
                                2010-04-30      -0.0081161      -0.0139653
                                2010-05-31      -0.0356510      -0.0280376
                                ------------------------------------------
        Return series of different holding periods
    '''
    idx = sub_portfolio_return.index.get_level_values('DATE')
    end = idx[-1]
    portfolio_return = pd.DataFrame(columns = holding_period, index = pd.date_range(start = start, end = end, freq = 'M'))
    
    for hd in holding_period:
        for month in portfolio_return.index:
            sub_portfolio_start = month - pd.offsets.MonthBegin(hd)
            sub_portfolio_data = sub_portfolio_return.loc[sub_portfolio_start:month]
            ret = []
            for i in range(hd):
                try:
                    portfolio_i_return_in_this_month = sub_portfolio_data.iloc[i,hd-1-i]
                    ret.append(portfolio_i_return_in_this_month)
                except IndexError:
                    print('%s %d has no data'%(str(month),hd))
                    
            ret = sum(ret)/hd
            portfolio_return.loc[month,hd] = ret
    
    portfolio_return.index.name = 'DATE'
    portfolio_return.columns.name = 'holding period'
    
    return portfolio_return
    

def get_singal_portfolio_return(stock_data, holding_period, start, signal_lag):
    '''
    calculate return series of one portfolio

    Parameters
    ----------
    stock_data : DataFrame 
        result of 'get_weight'.

    Returns
    -------
    Return series of this portfolio

    '''
    max_hd = max(holding_period)
    columns = [i for i in range(1,max_hd+1)]
    singal_stock_return = stock_data[columns]
    # get return of singal stock
    
    weight = stock_data['weights']
    # get weight of signal stock
    
    weighted_singal_stock_return = singal_stock_return.multiply(weight, axis=0)
    sub_portfolio_return = weighted_singal_stock_return.groupby(level = 'DATE').sum()
    # calculate return of each sub-portfolio
    
    portfolio_return = rolling_return_helper(sub_portfolio_return, start, holding_period)
    # calculate total portfolio return
    
    portfolio_return.index = portfolio_return.index.shift(1+signal_lag)
    portfolio_return = portfolio_return.iloc[:-(1+signal_lag),:]
    # shift return to current month
            
    return portfolio_return


def get_each_quantile_return(ret_data, holding_period, start, signal_lag, calculate_by_group = False):
    '''
    Get the return of portfolio on different quantile using equal weight

    Parameters
    ----------
    data : DataFrame
        result of 'get_weights'.

    Returns
    -------
    Returns of portfolio on each quantile

    '''
    data = ret_data.copy()
    
    max_hd = max(holding_period)
    columns = [i for i in range(1,max_hd+1)]
    
    data[columns] = data[columns].apply(lambda x: x * data['weights'])
    
    grouper = [data.index.get_level_values('DATE')]
    if calculate_by_group:
        grouper.append('group')
        
    grouper.append('factor_quantile')
    # generate a grouper to group data in order to calculate sub-portfolio return
    
    sub_portfolio_return = data.groupby(grouper)[columns].apply(lambda df: df.sum())
    # calcualte sub-portfolio return using equal weight in each quantile
        
    grouper = []
    if calculate_by_group:
        grouper.append(sub_portfolio_return.index.get_level_values('group'))
        
    grouper.append(sub_portfolio_return.index.get_level_values('factor_quantile'))
    # generate a new grouper in order to calculate portfolio return
    
    portfolio_return = sub_portfolio_return.groupby(grouper).apply(rolling_return_helper , start, holding_period)
    # calcualte portfolio return
    
    portfolio_return.index = portfolio_return.index.swaplevel(0,'DATE')
    
    portfolio_return.sort_index(inplace=True)
    # reorder index and sort index
    
    portfolio_return.index.set_levels(portfolio_return.index.get_level_values('DATE').shift(1+signal_lag), level = 0, inplace=True)
    
    portfolio_return.drop(index = portfolio_return.index.levels[0].unique()[-(1+signal_lag):], inplace = True)
    
    return portfolio_return
    


def get_market_return(data):
    '''
    Calculate the average return of stocks in a cross section as the market return

    Parameters
    ----------
    data : DataFrame
        output of 'get_quantile'.

    Returns
    -------
    Market return in each holding period.

    '''
    columns = [i for i in data.columns if 'RET' in i]
    grouper = [data.index.get_level_values['DATE']]
    
    mkt_return = data.groupby(grouper).mean()[columns]
    mkt_return = mkt_return.shift(freq = pd.offsets.MonthEnd(-1))
    mkt_return.columns = ['Current %s return'%i[12:] for i in columns]
    
    return mkt_return
    

def portfolio_return_analysis(portfolio_return, Print = True):
    
    '''
    generate return report

    Parameters
    ----------
    portfolio_return : DataFrame
        one group or multi group return series
        result of 'get_singal_portfolio_return' or 'get_each_quantile_return'

    Returns
    -------
    print and return the return table

    '''
    data = portfolio_return.copy()
    
    data = data.astype(float)
    
    level_num = data.index.nlevels
    
    if level_num>1:
        for i in range(1,level_num):
            data = data.unstack()
    
    report = data.describe().iloc[:4,:]
    #report.loc['mean'] = report.loc['mean']*2
    report.loc['skewness'] = data.skew()
    report.loc['kurtosis'] = data.kurt()
    
    t = stats.ttest_1samp(data, popmean = 0, nan_policy='omit' )
    report.loc['t value'] = t[0]
    report.loc['p value'] = t[1]
    
    report.loc['annualized sharpe ratio',:] = report.loc['mean',:]/report.loc['std',:] * np.sqrt(12)
        
    net_value = (1+data).cumprod()
    previous_high = net_value.cummax()
    drawdown = (net_value - previous_high)/previous_high 
    report.loc['max drawdown'] = drawdown.min()
    
    report.rename(index = {'count':'observations'}, inplace=True)
    if Print:
        print('return analysis:')
        print(report)
        print('\n')
    
    return report




def plot_net_value(portfolio_return):
    ## not support group plot yet.
    
    data = portfolio_return.copy()
    
    data = data.astype(float)
    
    level_num = data.index.nlevels
    
    if level_num>1:
        for i in range(1,level_num):
            data = data.unstack()
    
    holding_period = data.columns.get_level_values('holding period').unique()

    if level_num == 1:
        # single portfolio plot
        cum_return = (1+data).cumprod()
        cum_return.plot(grid=True, title = 'net value of long top and short bottom quantile portfolio in different holding period')
            
    else:
        # multi portfolio plot
        # first divide by holding period. Then divide by industry(if needed)
        nrows = len(holding_period)
        
        if level_num ==3:
            ncols = len(data.columns.get_level_values('group').unique())
        else:
            ncols = 1
        fig, ax = plt.subplots(nrows, ncols, figsize = (15,15), squeeze = False)
            
        cum_return = (1+data).cumprod()
        
        i=0
        j=0
        columns = pd.MultiIndex.from_product(data.columns.levels[:-1])
        for col in columns:
            
            col = col + (slice(None),)
            sub_data = cum_return.loc[:,col].copy()
            sub_data.columns = sub_data.columns.droplevel()
            
            sub_data.plot(ax=ax[i,j], grid = True).legend(loc='upper left')
            
            col = col[:-1]
            if len(col)==1:
                ax[i,j].title.set_text('net value of {} holding period'.format(col[0]))
            else:
                ax[i][j].title.set_text('net value of {} holding period {} industry'.format(col[0],col[1]))
            j = j+1 if j < ncols-1 else 0
            i = i+1 if j==0 else i
            
    plt.show()    
    
        
if __name__=='__main__':
    
    signal = input('Please input the serial number of signal:\n')
    
    start = input('Please enter the start time YYYYMMDD:\n')
    end = input('Please enter the end time YYYYMMDD\n')
    
    signal_lag = input('Please input signal lag (1 month is zero):\n')
    
    quantile = input('Please input quantile to divide stocks (For example, decile is 10):\n')
    
    exchanges = input('Please input the exchanges: [all/ NYSE/ NYSE MKT/ NASDAQ/ NYSE Arca]\n')
    
    weight_scheme = input('Please input the weight scheme: [value/equal]\n')
    
    exclude_penny_stock = input('do you want to drop penny stocks?[y/n]\n')
    
    holding_period = input('Please input holding period .You can input many number (For example: 1 2 means 1 month and 2 month):\n').split(' ')
    
    formation_period = input('Please input the formation period:\n')
    
    folder = [i for i in os.listdir(ENV_PATH) if signal in i][0]
    csv = [i for i in os.listdir(ENV_PATH+folder) if '.csv' in i and formation_period == i.split('-')[0]][0]
    
    exclude_penny_stock = True if exclude_penny_stock=='y' else False
    
    exchanges = [exchange_map[i] for i in exchanges.split(' ')] if exchanges!='all' else exchanges
    
    holding_period = [int(i) for i in holding_period]
    
    try: 
        quantile = int(quantile)
    except:
        quantile = [float(i) for i in quantile.split()]
        
    max_hd = max(holding_period)
    
    value_weighted = True if weight_scheme=='value' else False
    
    weight_scheme = 'value_weighted' if weight_scheme=='value' else 'equal_weighted'
    
    start = pd.to_datetime(start, format = '%Y%m%d')-pd.offsets.MonthEnd()
    # cause current time corresponding to next month return, need to calculate one month in advance
    end = pd.to_datetime(end, format = '%Y%m%d')
    # transfer string to datetime
    
    signal_lag = int(signal_lag)
    
    cal_start = start - pd.offsets.MonthBegin(max_hd + signal_lag)
    # ahead start date in order to calculate the first portfolio
    
    data = pd.read_csv(ENV_PATH+folder+'\\'+csv)
    
    data = get_forward_return( data, cal_start, end, signal_lag, exclude_penny_stock, value_weighted, exchanges, holding_period)
    
    start = max(data.index.get_level_values('DATE').min()+pd.offsets.MonthEnd(max_hd+signal_lag-1), start)
    
    data = get_quantile(data, quantile, groupby = None, quantize_by_group = False)

    weighted_data = get_weight( data, weight_scheme, quantile = 'all', hurdle = 0, calculate_by_group = False)
    
    quantile_return = get_each_quantile_return(weighted_data, holding_period, start, signal_lag, calculate_by_group = False)
    
    long_ret = quantile_return.xs(quantile, level='factor_quantile')
    short_ret = quantile_return.xs(1, level='factor_quantile')
    
    long_short_return = long_ret - short_ret
    
    print('Long top and short botton portfolio analysis:')
    
    long_short_return_analysis = portfolio_return_analysis(long_short_return)
    
    print('Net value of Long top short bottom portfolio:\n')
    
    plot_net_value(long_short_return)
        
    print('Quantile portfolio analysis:')
    
    quantile_return_analysis = portfolio_return_analysis(quantile_return)
    
    print('Net value of different quantile:\n')
    
    plot_net_value(quantile_return)
    
    
    
    
    

    
    
    
    
    
    
    
    
          
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    