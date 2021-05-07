import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.ensemble import RandomForestRegressor
from pipeline import AMLPipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

X, y = make_regression(random_state=1, n_samples=1000,
                       n_features=10, noise=100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42)

pipe = AMLPipeline([
    # ('imp', SimpleImputer(strategy=('mean', 'most_frequent'))),
    # ('imp2', KNNImputer(n_neighbors=(6, 7))),
    ('sca', StandardScaler()),
    ('sca2', Binarizer()),
    ('mod', DecisionTreeRegressor()),
    ('mod2', RandomForestRegressor())],
    metric=mean_absolute_error,
    save_performance=True)

pipe.fit(X_train, y_train)

report = pipe.validate(X_train, y_train, X_test, y_test, X_val, y_val)

# f'{report=}'.split('=')[0]
