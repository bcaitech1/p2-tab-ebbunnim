# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import os, sys, gc, warnings, random
import datetime
import dateutil.relativedelta

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
    print('#########complete generation###########')

    ##### corr col 제거 #####
    corr_cols = get_high_correlation_cols(df)
    print('corr 높은거 컬럼들: ', corr_cols)
    to_drop_high_corr = set(corr_cols)
    df.drop(to_drop_high_corr, axis=1, inplace=True)
    ##### corr col 제거 #####
    return df