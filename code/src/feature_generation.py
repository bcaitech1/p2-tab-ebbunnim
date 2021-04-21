# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import os, sys, gc, warnings, random
import datetime
from dateutil.relativedelta import relativedelta

# Data manipulation
import pandas as pd 
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Machine learning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from tqdm.notebook import trange, tqdm

# Custom
from utils import get_high_correlation_cols

pd.options.display.max_rows = 10000
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = 1000


def feature_generation_cumsum(data,year_month):
    df=data.copy()
    
    df['cumsum_total_by_cust_id']=df.groupby(['customer_id'])['total'].cumsum()
    df['cumsum_quantity_by_cust_id']=df.groupby(['customer_id'])['quantity'].cumsum()
    df['cumsum_price_by_cust_id']=df.groupby(['customer_id'])['price'].cumsum()

    df['cumsum_total_by_prod_id']=df.groupby(['product_id'])['total'].cumsum()
    df['cumsum_quantity_by_prod_id']=df.groupby(['product_id'])['quantity'].cumsum()
    df['cumsum_price_by_prod_id']=df.groupby(['product_id'])['price'].cumsum()
    
    df['cumsum_total_by_order_id']=df.groupby(['order_id'])['total'].cumsum()
    df['cumsum_quantity_by_order_id']=df.groupby(['order_id'])['quantity'].cumsum()
    df['cumsum_price_by_order_id']=df.groupby(['order_id'])['price'].cumsum()
    
    cols=[col for col in df.columns if 'cumsum' in col]

    return df,cols

def feature_generation_m_ym(data,year_month):
    df=data.copy()
      
    df['month']=df['order_date'].dt.month
    df['year_month']=df['order_date'].dt.strftime('%Y-%m')

    cols=['month','year_month']

    return df,cols

def feature_generation_time_series_diff(data,year_month):
    df=data.copy()
    
    df['order_ts']=df['order_date'].astype(np.int64)//1e9
    df['order_ts_diff']=df.groupby(['customer_id'])['order_ts'].diff()
    df['quantity_diff']=df.groupby(['customer_id'])['quantity'].diff()
    df['price_diff']=df.groupby(['customer_id'])['price'].diff()
    df['total_diff']=df.groupby(['customer_id'])['total'].diff()
    
    return df

def feature_generation_all(data,year_month):
    df=data.copy()

    df['cumsum_total_by_cust_id']=df.groupby(['customer_id'])['total'].cumsum()
    df['cumsum_quantity_by_cust_id']=df.groupby(['customer_id'])['quantity'].cumsum()
    df['cumsum_price_by_cust_id']=df.groupby(['customer_id'])['price'].cumsum()

    df['cumsum_total_by_prod_id']=df.groupby(['product_id'])['total'].cumsum()
    df['cumsum_quantity_by_prod_id']=df.groupby(['product_id'])['quantity'].cumsum()
    df['cumsum_price_by_prod_id']=df.groupby(['product_id'])['price'].cumsum()
    
    df['cumsum_total_by_order_id']=df.groupby(['order_id'])['total'].cumsum()
    df['cumsum_quantity_by_order_id']=df.groupby(['order_id'])['quantity'].cumsum()
    df['cumsum_price_by_order_id']=df.groupby(['order_id'])['price'].cumsum()
    
    df['order_ts']=df['order_date'].astype(np.int64)//1e9
    df['order_ts_diff']=df.groupby(['customer_id'])['order_ts'].diff()
    df['quantity_diff']=df.groupby(['customer_id'])['quantity'].diff()
    df['price_diff']=df.groupby(['customer_id'])['price'].diff()
    df['total_diff']=df.groupby(['customer_id'])['total'].diff()
    # feture 붙이기
    all_mean_col=all_mean_category(data,year_month)
    df=pd.merge(df,all_mean_col,on='customer_id',how='left')
    print(df.head())


    print('#########complete generation###########')

    ##### corr col 제거 #####
    # corr_cols = get_high_correlation_cols(df)
    # print('corr 높은거 컬럼들: ', corr_cols)
    # to_drop_high_corr = set(corr_cols)
    # df.drop(to_drop_high_corr, axis=1, inplace=True)
    ##### corr col 제거 #####
    return df

def all_mean_category(df,year_month):
    df = df.copy()
    first_buy = df.drop_duplicates('customer_id', keep='first')[['order_date','customer_id']].sort_values('customer_id').reset_index(drop=True)
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')

    #지금까지 한달에 평균 얼마씩 썼는지 구하기
    #처음 물건 산 뒤로 몇달 지났는지 계산하여 나눠줌
    first_buy['month']=first_buy['order_date'].map(lambda x:12*relativedelta(d, x).years+relativedelta(d, x).months).reset_index(drop=True)
    data_agg_sum = df.groupby('customer_id').sum().reset_index().sort_values('customer_id')
    first_buy['mean'] = data_agg_sum['total']/(first_buy['month']+1)
    return first_buy[['customer_id','mean']]


    def add_trend(train, test, year_month):
    train = train.copy()
    test = test.copy()

    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym_d = d - dateutil.relativedelta.relativedelta(months=1)

    train_window_ym = []
    test_window_ym = [] 
    for month_back in [1, 2, 3, 5, 7, 12, 20, 23]: # 1개월, 2개월, ... 20개월, 23개월 전 year_month 파악
        train_window_ym.append((prev_ym_d - dateutil.relativedelta.relativedelta(months = month_back)).strftime('%Y-%m'))
        test_window_ym.append((d - dateutil.relativedelta.relativedelta(months = month_back)).strftime('%Y-%m'))

    # aggregation 함수 선언
    agg_func = ['max','min','sum','mean','count','std','skew']

    # group by aggregation with Dictionary
    agg_dict = {
        'quantity': agg_func,
        'price': agg_func,
        'total': agg_func,
    }

    # general statistics for train data with time series trend
    for i, tr_ym in enumerate(train_window_ym):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['year_month'] >= tr_ym].groupby(['customer_id']).agg(agg_dict) # 해당 year_month 이후부터 모든 데이터에 대한 aggregation을 실시

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for level1, level2 in train_agg.columns:
            new_cols.append(f'{level1}-{level2}-{i}')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        if i == 0:
            train_data = train_agg
        else:
            train_data = train_data.merge(train_agg, on=['customer_id'], how='right')


    # general statistics for test data with time series trend
    for i, tr_ym in enumerate(test_window_ym):
        # group by aggretation 함수로 test 데이터 피처 생성
        test_agg = test.loc[test['year_month'] >= tr_ym].groupby(['customer_id']).agg(agg_dict)

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for level1, level2 in test_agg.columns:
            new_cols.append(f'{level1}-{level2}-{i}')

        test_agg.columns = new_cols
        test_agg.reset_index(inplace = True)
        
        if i == 0:
            test_data = test_agg
        else:
            test_data = test_data.merge(test_agg, on=['customer_id'], how='right')

    return train_data, test_data
