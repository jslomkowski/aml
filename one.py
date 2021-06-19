from pipeline import AMLPipeline
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import EqualFrequencyDiscretiser, EqualWidthDiscretiser
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

X, y = load_boston(return_X_y=True)
X = pd.DataFrame(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipe = AMLPipeline([
    ('disc1', EqualFrequencyDiscretiser()),
    ('disc2', EqualWidthDiscretiser()),
    ('model1', LinearRegression()),
    ('model2', RandomForestRegressor(n_estimators=(50, 100), criterion=('mse', 'mae')))
],
    mean_absolute_error, save_performance=True)

pipe.fit(X_train, y_train)
pipe.validate(X_train, y_train, X_test, y_test)
