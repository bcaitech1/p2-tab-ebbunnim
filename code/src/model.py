from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np

import os
from pathlib import Path
from matplotlib import pyplot as plt

# -- settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# TabNetPretrainer
unsupervised_model = TabNetPretrainer(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-3),
    mask_type='entmax', # "sparsemax",
    device_name='cuda'
)

# unsupervised_model.fit(
#     X_train=X_train,
#     eval_set=[X_valid],
#     pretraining_ratio=0.8,
# )

clf = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-3),
    scheduler_params={"step_size":7, # how to use learning rate scheduler
                      "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax', # This will be overwritten if using pretrain model,
    device_name='cuda'

)

# clf.fit(
#     X_train=X_train, y_train=y_train,
#     eval_set=[(X_train, y_train), (X_valid, y_valid)],
#     eval_name=['train', 'valid'],
#     eval_metric=['auc'],
#     from_unsupervised=unsupervised_model
# )

