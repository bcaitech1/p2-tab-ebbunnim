# Suppress warnings 
import warnings

warnings.filterwarnings('ignore')

import datetime
import gc
import os
import random
import sys
import warnings

import dateutil.relativedelta
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm.notebook import tqdm, trange

# Feature generation
from feature_generation import (add_seasonality, add_trend,
                                feature_generation_all,
                                feature_generation_cumsum,
                                feature_generation_m_ym,
                                feature_generation_time_series_diff)

pd.options.display.max_rows = 10000
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = 1000

TOTAL_THRES = 300

def generate_label(df, year_month, total_thres=TOTAL_THRES, print_log=False):
    df = df.copy()

    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    df.reset_index(drop=True, inplace=True)

    cust = df[df['year_month']<year_month]['customer_id'].unique()
    df = df[df['year_month']==year_month]
    
    label = pd.DataFrame({'customer_id':cust})
    label['year_month'] = year_month
    
    grped = df.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    
    label = label.merge(grped, on=['customer_id','year_month'], how='left')
    label['total'].fillna(0.0, inplace=True)
    label['label'] = (label['total'] > total_thres).astype(int)

    label = label.sort_values('customer_id').reset_index(drop=True)
    if print_log: print(f'{year_month} - final label shape: {label.shape}')
    
    return label

def feature_preprocessing(train, test, features, do_imputing=True):
    x_tr = train.copy()
    x_te = test.copy()
    
    cate_cols = []

    for f in features:
        if x_tr[f].dtype.name == 'object':
            cate_cols.append(f)
            le = LabelEncoder()

            le.fit(list(x_tr[f].values) + list(x_te[f].values))
            
            x_tr[f] = le.transform(list(x_tr[f].values))
            
            x_te[f] = le.transform(list(x_te[f].values))

    print('categorical feature:', cate_cols)

    if do_imputing:
        imputer = SimpleImputer(strategy='median')

        x_tr[features] = imputer.fit_transform(x_tr[features])
        x_te[features] = imputer.transform(x_te[features])
    
    return x_tr, x_te

def feature_engineering_base(df, year_month):
    """
        feature generation 필요 없음.
    """
    df = df.copy()
    
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]
    
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']]
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    
    agg_func = ['mean','max','min','sum','count','std','skew']
    all_train_data = pd.DataFrame()
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_func)

        new_cols = []
        for col in train_agg.columns.levels[0]:
            for stat in train_agg.columns.levels[1]:
                new_cols.append(f'{col}-{stat}')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        all_train_data = all_train_data.append(train_agg)
    
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    
    test_agg = test.groupby(['customer_id']).agg(agg_func)
    test_agg.columns = new_cols
    
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')

    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, all_train_data['label'], features

def feature_engineering_cumsum(data, year_month):
    """
        feature_generation_cumsum 필요
    """
    df = data.copy()
    
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    
    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]
    
    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']]
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    
    # group by aggregation 함수 선언
    agg_func = ['mean','max','min','sum','count','std','skew']
    all_train_data = pd.DataFrame()
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        
        train,cols=feature_generation_cumsum(train,year_month)
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id'])[cols].agg(agg_func)

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for col in train_agg.columns.levels[0]:
            for stat in train_agg.columns.levels[1]:
                new_cols.append(f'{col}-{stat}')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        all_train_data = all_train_data.append(train_agg)
    
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    
    # group by aggretation 함수로 test 데이터 피처 생성
    test,cols=feature_generation_cumsum(test,year_month)
    test_agg = test.groupby(['customer_id'])[cols].agg(agg_func)
    test_agg.columns = new_cols
    
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')
    
    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, all_train_data['label'], features

def feature_engineering_nunique(data, year_month):
    """
        feature generation 내부에 구현되어 있음
    """
    df = data.copy()
    
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    
    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]
    
    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']]
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    
    # group by aggregation 함수 선언
    all_train_data = pd.DataFrame()
    
    # for feature generation
    cols=['order_id','product_id']
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id'])[cols].agg(['nunique'])

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for col in train_agg.columns.levels[0]:
            for stat in train_agg.columns.levels[1]:
                new_cols.append(f'{col}-{stat}')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        all_train_data = all_train_data.append(train_agg)
    
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    
    # group by aggretation 함수로 test 데이터 피처 생성
    test_agg = test.groupby(['customer_id'])[cols].agg(['nunique'])
    test_agg.columns = new_cols
    
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')

    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, all_train_data['label'], features

def feature_engineering_m_ym(data, year_month):
    """
        feature_generation_m_ym 필요
    """
    df = data.copy()
    
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    
    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]
    
    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']]
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    
    # group by aggregation 함수 선언
    all_train_data = pd.DataFrame()
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        
        train,cols=feature_generation_m_ym(train,year_month)
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id'])[cols].agg([lambda x : x.value_counts().index[0]])

        train_agg.columns = ['month-mode','year_month-mode']
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        all_train_data = all_train_data.append(train_agg)
    
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    
    # group by aggretation 함수로 test 데이터 피처 생성
    test,cols=feature_generation_m_ym(test,year_month)
    test_agg = test.groupby(['customer_id'])[cols].agg([lambda x : x.value_counts().index[0]])
    test_agg.columns = ['month-mode','year_month-mode']
    
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')
    
    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, all_train_data['label'], features

def feature_engineering_time_series_diff(data, year_month):
    """
        feature_generation_time_series_diff 필요
    """
    df = data.copy()
    
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    
    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]
    
    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']]
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    
    # group by aggregation 함수 선언
    agg_func = ['mean','max','min','sum','count','std','skew']

    agg_dict = {
        'order_ts':['first','last'],
        'order_ts_diff':agg_func,
        'quantity_diff':agg_func,
        'price_diff':agg_func,
        'total_diff':agg_func,
    }
    
    all_train_data = pd.DataFrame()
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train=feature_generation_time_series_diff(train,year_month)
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_dict)

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for col in agg_dict.keys():
            for stat in agg_dict[col]:
                new_cols.append(f'{col}-{stat}')
                
        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        all_train_data = all_train_data.append(train_agg)
    
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    
    # group by aggretation 함수로 test 데이터 피처 생성
    test=feature_generation_time_series_diff(test,year_month)
    test_agg = test.groupby(['customer_id']).agg(agg_dict)
    test_agg.columns = new_cols
    
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')
    
    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, all_train_data['label'], features

def feature_engineering_all(data, year_month):
    """
        feature_generation_all 필요
    """
    df = data.copy()
    
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    
    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]


    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']]
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    
    print('#########complete labeling###########')

    # group by aggregation 함수 선언
    agg_func = ['mean','sum','count','skew']

    agg_dict = {
        'order_ts':['first','last'],
        'order_ts_diff':agg_func,
        'quantity_diff':agg_func,
        'price_diff':agg_func,
        'total_diff':agg_func,
        'cumsum_total_by_cust_id':agg_func,
        'cumsum_quantity_by_cust_id':agg_func, ### corr에 의해 drop해봤으나 안한게 낫다.
        'cumsum_price_by_cust_id':agg_func,
        'cumsum_total_by_prod_id':agg_func,
        'cumsum_quantity_by_prod_id':agg_func,
        'cumsum_price_by_prod_id':agg_func,
        'cumsum_total_by_order_id':agg_func,
        'cumsum_quantity_by_order_id':agg_func,
        'cumsum_price_by_order_id':agg_func,
        'mean':agg_func,
    }
    
    all_train_data = pd.DataFrame()
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train=feature_generation_all(train,year_month)
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_dict)

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for col in agg_dict.keys():
            for stat in agg_dict[col]:
                new_cols.append(f'{col}-{stat}')
                
        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        all_train_data = all_train_data.append(train_agg)
    
    test=feature_generation_all(test,year_month)
    test_agg = test.groupby(['customer_id']).agg(agg_dict)
    test_agg.columns = new_cols

    ############## tracking trend ##################
    t1,t2=add_trend(train, test, year_month)
    t3,t4=add_seasonality(train, test, year_month)

    # add trend (merge feature)
    all_train_data=all_train_data.merge(t1,on=['customer_id'], how='left')
    all_train_data=all_train_data.merge(t3,on=['customer_id'], how='left')
    print('=====complete trend merge(train)======')
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    print('features: ',features)

    #### feature selection ####
    # X=all_train_data.drop(columns=['customer_id', 'label', 'year_month'])
    # Y=train_label.drop(columns=['customer_id','year_month'])
    # print('123213123123123123123')
    # print(train_label)
    # features = feature_selector.fit(X,Y)
    # filtered_features= features.columns[list(features.k_feature_idx_)]
    # print('filtered features: ',filtered_features)
    ################################################

    # group by aggretation 함수로 test 데이터 피처 생성
    # add trend (merge feature)
    test_agg=test_agg.merge(t2,on=['customer_id'],how='left')
    test_agg=test_agg.merge(t4,on=['customer_id'],how='left')
    print('=====complete trend merge(test)======')
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')
    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    return x_tr, x_te, all_train_data['label'], features
