# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:58:15 2020

@author: Gao Haocheng
"""

import pandas as pd

ENV_PATH =  "C:\\Users\\apply\\Desktop\\OneDrive - National University of Singapore\\Stock_CSV\\"

month_std_res = pd.read_csv(ENV_PATH+'206-Idiosyncratic Risk\\1-0-1.csv')
month_std_res['DATE'] = pd.to_datetime(month_std_res['DATE'], format = '%Y-%m-%d')

def get_avg_std(data):
    
    data.set_index('DATE',inplace=True)
    grouper = data.groupby(by=['PERMNO'])
    
    data.index= data.index - pd.offsets.MonthBegin() + pd.offsets.MonthEnd()
    
    avg_data = []
    
    for name, sub_data in grouper:
        index = pd.date_range(start = sub_data.index.min(), end = sub_data.index.max(), freq='M')
        sub_data = sub_data.reindex(index)
        
        sub_data['avg'] = sub_data['SIGNAL'].rolling(window=11,min_periods=5).mean()
        # calculate in previous 11 month
        
        avg_data.append(sub_data)
    
    avg_data = pd.concat(avg_data,axis=0)
    
    avg_data.index = avg_data.index + pd.offsets.MonthEnd(1)#skip current month
    
    avg_data.dropna(inplace=True)
    
    return avg_data[['PERMNO','avg']].rename(columns={'avg':'SIGNAL'})

avg = get_avg_std(month_std_res)
avg.to_csv(ENV_PATH+'199-Idiosyncratic Risk, Distant\\11-1-1.csv')
