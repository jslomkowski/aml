import pandas as pd
import pytest
from aml import AMLGridSearchCV
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from aml.models_template import aml_basic_regressors


@pytest.fixture(scope="session")
def scenario_without_params():
    """This is to create pipeline test scenario one without params.

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


@pytest.fixture(scope="session")
def scenario_with_class_import():
    # ! TODO
    """This is to create pipeline test scenario one without params.

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
        aml_basic_regressors
    ])

    param_grid = {}

    aml = AMLGridSearchCV(pipeline, param_grid)
    return aml, pipeline, param_grid
