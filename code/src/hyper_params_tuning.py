import optuna
from inference import make_lgb_oof_prediction
from importlib import import_module
from feature_engineering import generate_label
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score


def objective(trial):
    lgb_params = {
        'objective': 'binary', # 이진 분류
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 2, 256), # num_leaves 값을 2-256까지 정수값 중에 사용
        'max_bin': trial.suggest_int('max_bin', 128, 256), # max_bin 값을 128-256까지 정수값 중에 사용
        # min_data_in_leaf 값을 10-40까지 정수값 중에 사용
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 40),
        # 피처 샘플링 비율을 0.4-1.0까지 중에 uniform 분포로 사용
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        # 데이터 샘플링 비율을 0.4-1.0까지 중에 uniform 분포로 사용
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        # 데이터 샘플링 횟수를 1-7까지 정수값 중에 사용
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'n_estimators': 10000, # 트리 개수
        'early_stopping_rounds': 100,
        # L1 값을 1e-8-10.0까지 로그 uniform 분포로 사용
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        # L2 값을 1e-8-10.0까지 로그 uniform 분포로 사용
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'seed': 0, # ##### 이거 원래 SEED로 넘겨야 되는데.....
        'verbose': -1,
        'n_jobs': -1,    
    }
    # 데이터 파일 읽기
    data_dir = '/opt/ml/code/input' 
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])
    train, test, y, features = getattr(import_module("feature_engineering"), 'feature_engineering_time_series_diff')(data, '2011-12')
    
    # oof prediction 함수 호출해서 out of fold validation 예측값을 얻어옴
    y_oof, test_preds, fi = make_lgb_oof_prediction(train, y, test, features, model_params=lgb_params)
    
    # Validation 스코어 계산
    label = generate_label(data, '2011-11')['label']
    val_auc = roc_auc_score(label, y_oof)
    
    return val_auc

def get_best_params():
    label = generate_label(data, '2011-11')['label']
    study = optuna.create_study(direction='maximize')
    study.optimize(objective,n_trials=50) # 10회 동안 하이퍼 파라미터 탐색
    print(study.best_params)
    return study.best_params

if __name__ == '__main__':
    # 인자 파서 선언
    parser = argparse.ArgumentParser()
    
    # baseline 모델 이름 인자로 받아서 model 변수에 저장
    parser.add_argument('--seed', type=int, default=0, help="base seed is 42")
    parser.add_argument('--ym', type=str, default='2011-12', help="add target year_month to predict, base is 2011-12")
    parser.add_argument('--engineering', type=str, default='feature_engineering_time_series_diff', help="base type is feature engineering type")
    parser.add_argument('--tuning', type=bool, default=True, help="choose true if you want hyper params tuning with optuna, else choose false")

    args = parser.parse_args()

    data_dir = '/opt/ml/code/input' 
    model_dir = '/opt/ml/code/model' 
    output_dir = '/opt/ml/code/output' 

    # 데이터 파일 읽기
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])

    # 예측할 연월 & 시드 설정
    year_month = args.ym
    seed=args.seed

    get_best_params()