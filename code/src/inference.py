# Suppress warnings 
import warnings

warnings.filterwarnings('ignore')

import argparse
import datetime
import gc
import os
import random
import sys
from importlib import import_module

import dateutil.relativedelta
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
# Weight Ensemble
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
# Machine learning
from sklearn.preprocessing import LabelEncoder

from feature_engineering import (feature_engineering_base,
                                 feature_engineering_cumsum,
                                 feature_engineering_m_ym,
                                 feature_engineering_nunique,
                                 feature_engineering_time_series_diff,
                                 generate_label)
# Custom library
from utils import seed_everything

TOTAL_THRES = 300 # 구매액 임계값
def make_lgb_oof_prediction(train, y, test, features, categorical_features='auto', model_params=None, folds=10):
    # 시드 고정
    seed_everything(seed)

    x_train = train[features]
    x_test = test[features]
    
    test_preds = np.zeros(x_test.shape[0])
    
    y_oof = np.zeros(x_train.shape[0])
    
    score = 0
    
    fi = pd.DataFrame()
    fi['feature'] = features
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # LightGBM 데이터셋 선언
        dtrain = lgb.Dataset(x_tr, label=y_tr)
        dvalid = lgb.Dataset(x_val, label=y_val)
        
        # LightGBM 모델 훈련
        clf = lgb.train(
            model_params,
            dtrain,
            valid_sets=[dtrain, dvalid], # Validation 성능을 측정할 수 있도록 설정
            categorical_feature=categorical_features,
            verbose_eval=200
        )

        # Validation 데이터 예측
        val_preds = clf.predict(x_val)
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 측정
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(x_test) / folds
        
        # 폴드별 피처 중요도 저장
        fi[f'fold_{fold+1}'] = clf.feature_importance()

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력
        
    # 폴드별 피처 중요도 평균값 계산해서 저장 
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return y_oof, test_preds, fi

def make_cat_oof_prediction(train, y, test, features, categorical_features=None, model_params=None, folds=10):
    x_train = train[features]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')
        
        # CatBoost 모델 훈련
        clf = CatBoostClassifier(**model_params)
        clf.fit(x_tr, y_tr,
                eval_set=(x_val, y_val), # Validation 성능을 측정할 수 있도록 설정
                cat_features=categorical_features,
                use_best_model=True,
                verbose=True)
        
        # Validation 데이터 예측
        val_preds = clf.predict_proba(x_val)[:,1]
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 출력
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict_proba(x_test)[:,1] / folds

        # 폴드별 피처 중요도 저장
        fi[f'fold_{fold+1}'] = clf.feature_importances_
        
        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 평균 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력
        
    # 폴드별 피처 중요도 평균값 계산해서 저장
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return y_oof, test_preds, fi

def make_xgb_oof_prediction(train, y, test, features, model_params=None, folds=10):
    x_train = train[features]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')
        
        # XGBoost 데이터셋 선언
        dtrain = xgb.DMatrix(x_tr, label=y_tr)
        dvalid = xgb.DMatrix(x_val, label=y_val)
        
        # XGBoost 모델 훈련
        clf = xgb.train(
            model_params,
            dtrain,
            num_boost_round=10000, # 트리 개수
            evals=[(dtrain, 'train'), (dvalid, 'valid')],  # Validation 성능을 측정할 수 있도록 설정
            verbose_eval=200,
            early_stopping_rounds=100
        )
        
        # Validation 데이터 예측
        val_preds = clf.predict(dvalid)
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 출력
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(xgb.DMatrix(x_test)) / folds

        # 폴드별 피처 중요도 저장
        fi_tmp = pd.DataFrame.from_records([clf.get_score()]).T.reset_index()
        fi_tmp.columns = ['feature',f'fold_{fold+1}']
        fi = pd.merge(fi, fi_tmp, on='feature')

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 평균 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력
        
    # 폴드별 피처 중요도 평균값 계산해서 저장
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return y_oof, test_preds, fi

if __name__ == '__main__':
    # 인자 파서 선언
    parser = argparse.ArgumentParser()
    
    # baseline 모델 이름 인자로 받아서 model 변수에 저장
    parser.add_argument('--seed', type=int, default=0, help="base seed is 42")
    parser.add_argument('--ym', type=str, default='2011-12', help="add target year_month to predict, base is 2011-12")
    parser.add_argument('--engineering', type=str, default='feature_engineering_all', help="base type is feature engineering type")
    parser.add_argument('--tuning', type=bool, default=True, help="choose true if you want hyper params tuning with optuna, else choose false")
    parser.add_argument('--ensemble', type=bool, default=True, help="choose true if you want to ensemble model, else choose false")

    args = parser.parse_args()

    data_dir = '/opt/ml/code/input' 
    model_dir = '/opt/ml/code/model' 
    output_dir = '/opt/ml/code/output' 

    # 데이터 파일 읽기
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])

    # 예측할 연월 & 시드 설정
    year_month = args.ym
    seed=args.seed

    # hyper params tuning 안할때 base params
    lgb_params = {
        'objective': 'binary', # 이진 분류
        'boosting_type': 'gbdt', # default : gbdt
        'metric': 'auc', # 평가 지표 설정
        'feature_fraction': 0.8, # 피처 샘플링 비율
        'bagging_fraction': 0.8, # 데이터 샘플링 비율
        'bagging_freq': 1,
        'n_estimators': 10000, # 트리 개수
        'early_stopping_rounds': 30,
        'seed': args.seed,
        'verbose': -1,
        'n_jobs': -1,    
    }

    cat_params = {
        'n_estimators': 10000, # 트리 개수
        'learning_rate': 0.07, # 학습률
        'eval_metric': 'AUC', # 평가 지표 설정
        'loss_function': 'Logloss', # 손실 함수 설정
        'random_seed': args.seed,
        'metric_period': 100,
        'od_wait': 100, # early stopping round
        'depth': 6, # 트리 최고 깊이
        'rsm': 0.8, # 피처 샘플링 비율
    }

    xgb_params = {
        'objective': 'binary:logistic', # 이진 분류
        'learning_rate': 0.1, # 학습률
        'max_depth': 6, # 트리 최고 깊이
        'colsample_bytree': 0.8, # 피처 샘플링 비율
        'subsample': 0.8, # 데이터 샘플링 비율
        'eval_metric': 'auc', # 평가 지표 설정
        'seed': args.seed,
    } 


    # 피처 엔지니어링 실행
    train, test, y, features = getattr(import_module("feature_engineering"), args.engineering)(data, year_month)
    
    if args.ensemble:
        # oof_xgb, xgb_pred, fi = make_xgb_oof_prediction(train, y, test, features, model_params=xgb_params)
        oof_lgb, lgb_pred, fi = make_lgb_oof_prediction(train, y, test, features, model_params=lgb_params)
        oof_cat, cat_pred, fi = make_cat_oof_prediction(train, y, test, features, model_params=cat_params)
        test_preds = lgb_pred*0.4 + cat_pred*0.6
    else:
        y_oof, test_preds, fi = make_cat_oof_prediction(train, y, test, features, model_params=cat_params)

    # 테스트 결과 제출 파일 읽기
    sub = pd.read_csv(data_dir + '/sample_submission.csv')
    
    # 테스트 예측 결과 저장
    sub['probability'] = test_preds
    print(sub['probability'].head())
    
    os.makedirs(output_dir, exist_ok=True)
    # 제출 파일 쓰기
    sub.to_csv(os.path.join(output_dir , 'output.csv'), index=False) # /output.csv 라고 / 하면 안됨

    # 제출
    # submit.py실행하기 


