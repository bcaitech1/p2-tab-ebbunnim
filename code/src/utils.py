import os
import random

import numpy as np
import torch
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)


# 시드 고정 함수
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# 평가 지표 출력 함수
def print_score(label, pred, prob_thres=0.5):
    print('Precision: {:.5f}'.format(precision_score(label, pred>prob_thres)))
    print('Recall: {:.5f}'.format(recall_score(label, pred>prob_thres)))
    print('F1 Score: {:.5f}'.format(f1_score(label, pred>prob_thres)))
    print('ROC AUC Score: {:.5f}'.format(roc_auc_score(label, pred)))

# 시드 고정 함수 - torch 사용시
def seed_everything_for_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_high_correlation_cols(df, corrThresh=0.9):
    numeric_cols = df._get_numeric_data().columns
    corr_matrix = df.loc[:, numeric_cols].corr().abs()
    print(corr_matrix)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corrThresh)]
    return to_drop

