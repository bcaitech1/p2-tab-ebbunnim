# Readme

# Description

This repo was generated for participating in the [upstage contest](http://boostcamp.stages.ai/competitions/3/overview/description). I got 0.8598 AUC, conducting ensemble catboost and lgb model from tabular data.

- Goal : `Binary classification`    
Using log data(2009/12~2011/12) from 5914 users, predict the probability of each customer's total purchases exceeding 300 on December 2011. 

- Module execution : Inference > feature engineering > feature generation
    - `inference.py`     
    With parsed arguments, print out `Out Of Fold Validation Score` by using StratifiedKFold from sklearn.model_selection. 
    - `feature_generation.py`         
    By adding or subtracting required columns, manipulate pandas dataframe.
    - `feature_engineering`    
        - Execute functions : **generate label**, **feature preprocess**, and **feature generation**
        - By using the aggregation function, grow features or merge features 

# Installation

`pip install -r requirements.txt` at ./code dir.
```txt
catboost==0.24.4
lightgbm==3.1.1
matplotlib==3.1.3
numpy==1.19.5
pandas==1.1.5
scikit-learn==0.24.1
seaborn==0.11.1
xgboost==1.3.3
```

# Example
[inference.py](./code/src/inference.py)    

From arg parser, you can choose one from various options.

```python
parser.add_argument('--seed', type=int, default=0, help="base seed is 0")
parser.add_argument('--ym', type=str, default='2011-12', help="add target year_month to predict, base is 2011-12")
parser.add_argument('--engineering', type=str, default='feature_engineering_all', help="choose feature engineering type")
parser.add_argument('--ensemble', type=bool, default=True, help="choose true if you want to ensemble model, else choose false")

```
```python
train, test, y, features = getattr(import_module("feature_engineering"), args.engineering)(data, year_month)

if args.ensemble:
    oof_xgb, xgb_pred, fi = make_xgb_oof_prediction(train, y, test, features, model_params=xgb_params)
    oof_lgb, lgb_pred, fi = make_lgb_oof_prediction(train, y, test, features, model_params=lgb_params)
    oof_cat, cat_pred, fi = make_cat_oof_prediction(train, y, test, features, model_params=cat_params)
    ...
```


# Contraints
- `Input directory doesn't exist!`
    - `train.csv` and `sample_submission.csv` was not uploaded. But the column info was opened below.
- Columns of train data 
![데이터정보](https://user-images.githubusercontent.com/46434838/115820033-03e07700-a43b-11eb-8b91-e30d446987a9.png)


# Improvements
- ~~Things to improve.~~
- Add Feature selection ([sample](./code/src/feature_selection.py))
- Using Tabnet architecture. ([sample1](./code/src/model.py), [sample2](./code/src/train.py))