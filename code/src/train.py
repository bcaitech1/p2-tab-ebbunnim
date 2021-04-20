# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta
import argparse
import torch

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

# Custom library
from utils import seed_everything_for_torch, print_score
from features import generate_label, feature_engineering1
from model import unsupervised_model, clf
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier


TOTAL_THRES = 300 # 구매액 임계값
SEED = 42 # 랜덤 시드
seed_everything_for_torch(SEED) # 시드 고정


data_dir = './code/input' # os.environ['SM_CHANNEL_TRAIN']
model_dir = './code/model' # os.environ['SM_MODEL_DIR']
output_dir = './code/output' # os.environ['SM_OUTPUT_DATA_DIR']

######tabnet training######
def make_tabnet_oof_prediction(train, y, test, features, categorical_features='auto', folds=10):
    
    ####################MLFLOW###########################
    import mlflow
    HOST = "http://localhost"
    mlflow.set_tracking_uri(HOST+":6006/")
    mlflow.start_run()
    ####################MLFLOW###########################

    x_train = train[features]
    x_test = test[features]

    # # 모델 호출
    # unsupervised_model.fit(
    #     X_train=x_train.values, # values는 np.array랑 똑같은 역할
    #     eval_set=[x_test.values],
    #     max_epochs=1000 , patience=50,
    #     batch_size=2048, virtual_batch_size=128,
    #     drop_last=False,
    #     pretraining_ratio=0.8,
    # )

    clf = TabNetClassifier(
        n_d=64, n_a=64, n_steps=5,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":20,
                            "gamma":0.95},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax', 
        lambda_sparse=1e-4,
        device_name='auto',
    )  

    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros((x_test.shape[0])) # 21 - feature dimension으로 설정
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros((x_train.shape[0]))
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = np.array(x_train.loc[tr_idx, features]), np.array(x_train.loc[val_idx, features])
        y_tr, y_val = np.array(y[tr_idx]),np.array(y[val_idx])


        # clf.fit(
        #     X_train=x_tr, y_train=y_tr,
        #     eval_set=[(x_tr, y_tr), (x_val, y_val)],
        #     eval_name=['train', 'valid'],
        #     batch_size=1024, virtual_batch_size=128,
        #     eval_metric=['auc'],
        #     max_epochs=1000 , patience=50,
        #     from_unsupervised=unsupervised_model,
        #     num_workers=4
        # )

        clf.fit(
        x_tr, y_tr,
        eval_set=[(x_val, y_val)],
        max_epochs=1000 , patience=20,
        batch_size=1024, virtual_batch_size=128,
        )


        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # Validation 데이터 예측
        val_preds = clf.predict_proba(x_val)[:,1]

        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 측정
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict_proba(x_test.values)[:,1] / folds
        
        # 폴드별 피처 중요도 저장
        fi[f'fold_{fold+1}'] = clf.feature_importances_

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력
        
    ####################MLFLOW###########################
    mlflow.log_param("folds", folds)
    # for k,v in model_params.items():
    #     mlflow.log_param(k, v)

    mlflow.log_metric("Mean AUC", score)
    mlflow.log_metric("OOF AUC", roc_auc_score(y, y_oof))
    mlflow.end_run()
    ####################MLFLOW###########################

    # 폴드별 피처 중요도 평균값 계산해서 저장 
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return y_oof, test_preds, fi


if __name__ == '__main__':

    # 데이터 파일 읽기
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 예측할 연월 설정
    year_month = '2011-12'
    
    # 피처 엔지니어링 실행
    train, test, y, features = feature_engineering1(data, year_month)
    # Cross Validation Out Of Fold로 LightGBM 모델 훈련 및 예측
    y_oof, test_preds, fi = make_tabnet_oof_prediction(train, y, test, features)
    
    # 테스트 결과 제출 파일 읽기
    sub = pd.read_csv(data_dir + '/sample_submission.csv')
    
    # 테스트 예측 결과 저장
    sub['probability'] = test_preds
    print(sub['probability'].head())
    
    os.makedirs(output_dir, exist_ok=True)
    # 제출 파일 쓰기
    sub.to_csv(os.path.join(output_dir , 'output.csv'), index=False) # /output.csv 라고 / 하면 안됨

    # 자동 제출
    # output_dir= '/opt/ml/code/output'
    # user_key='Bearer 5cc45800a3739a5e62f5975948d1142853d88723'
    # submit(user_key, os.path.join(output_dir, 'output.csv'))