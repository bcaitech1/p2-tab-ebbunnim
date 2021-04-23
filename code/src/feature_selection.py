from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score

feature_selector = ExhaustiveFeatureSelector(
            RandomForestClassifier(n_jobs=-1),
            min_features=50,
            max_features=200,
            scoring='roc_auc',
            print_progress=True,
            cv=10)