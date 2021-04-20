# 정리



1. tabnet 정리
2. validation 전략 : time-series로 바꿔볼까?
3. 파라미터 튜닝
https://neptune.ai/blog/lightgbm-parameters-guide
4. 모델 전처리 및 eda로 피처삽입
5. 환경 최적화 



---

준비물: 터미널, tmux, mlflow

우선 할당 받은 서버를 터미널로 접속하신 다음에 다음과 같이 두개를 설치해주세요!

```
apt-get install tmux
pip install mlflow
```

tmux 같은 경우에는 background 작업을 돌릴때 사용되며, 터미널을 꺼도 돌아야 하는 작업과 같은 것들을 세팅할 때 아주 유용합니다!

우선 tmux session 실행

```
tmux new-session -s mlflow
```

화면이 전환되면, mlflow ui 실행 ( 원래는 tensorboard 용 포트인데, 저희는 일단 빌려쓰겠습니다 ㅎㅎ)

```
mlflow ui --backend-store-uri sqlite:///store.db --host 0.0.0.0 --port 6006
```

그리고 tmux 화면을 빠져 나오는 방식은 ctrl+b 그리고 d 를 박자에 잘 맞춰서 눌러주세요! 이게 조금 어려우시면 그냥 터미널을 끄셔도 상관은 없습니다.

그 다음 코드에 mlflow tracking 적용!

저는 inference 코드를 수정하여 다음과 같이 추가하였습니다.
```python
def make_lgb_oof_prediction(train, y, test, features, categorical_features='auto', model_params=None, folds=10):
    ####################MLFLOW###########################
    import mlflow
    HOST = "http://localhost"
    mlflow.set_tracking_uri(HOST+":6006/")
    mlflow.start_run()
    ####################MLFLOW###########################
    
    x_train = train[features]
    x_test = test[features]
    
 중간코드 생략.............
        
    print(f"\nMean AUC = {score}") # 폴드별 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력
    
    ####################MLFLOW###########################
    mlflow.log_param("folds", folds)
    for k,v in model_params.items():
        mlflow.log_param(k, v)

    mlflow.log_metric("Mean AUC", score)
    mlflow.log_metric("OOF AUC", roc_auc_score(y, y_oof))
    mlflow.end_run()
    ####################MLFLOW###########################

        
    # 폴드별 피처 중요도 평균값 계산해서 저장 
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return y_oof, test_preds, fi
```

