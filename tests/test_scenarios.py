from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import EqualFrequencyDiscretiser, EqualWidthDiscretiser
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from aml import AMLGridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd


def scenario_one():
    """This is to create pipeline without params.

    Returns:
        [aml pipeline, pipeline and param_grid]: self explanatory
    """
    X, y = load_boston(return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pipeline = Pipeline([
        ('disc1', EqualFrequencyDiscretiser()),
        ('disc2', EqualWidthDiscretiser()),
        ('model1', LinearRegression()),
        ('model2', RandomForestRegressor())
    ])

    param_grid = {}

    aml = AMLGridSearchCV(pipeline, param_grid)
    return aml, pipeline, param_grid
