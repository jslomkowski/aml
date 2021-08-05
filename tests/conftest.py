import pytest
from aml import AMLGridSearchCV
from aml.models_template import aml_basic_regressors
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


@pytest.fixture(scope="session")
def scenario_without_params():
    """This is to create pipeline test scenario one without params.

    Returns:
        [aml pipeline, pipeline and param_grid]: self explanatory
    """

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
def scenario_with_default_models_template():
    # ! TODO
    """This is to create pipeline test scenario one without params.

    Returns:
        [aml pipeline, pipeline and param_grid]: self explanatory
    """

    pipeline = Pipeline([
        ('disc1', EqualFrequencyDiscretiser()),
        ('disc2', EqualWidthDiscretiser()),
        aml_basic_regressors
    ])

    param_grid = {}

    aml = AMLGridSearchCV(pipeline, param_grid)
    return aml, pipeline, param_grid


@pytest.fixture(scope="session")
def scenario_with_custom_models_template():
    # ! TODO
    """This is to create pipeline test scenario one without params.

    Returns:
        [aml pipeline, pipeline and param_grid]: self explanatory
    """

    regressors = [
        ('model1', LinearRegression()),
        ('model2', RandomForestRegressor())
    ]

    pipeline = Pipeline([
        ('disc1', EqualFrequencyDiscretiser()),
        ('disc2', EqualWidthDiscretiser()),
        regressors
    ])

    param_grid = {}

    aml = AMLGridSearchCV(pipeline, param_grid)
    return aml, pipeline, param_grid


@pytest.fixture(scope="session")
def scenario_with_grid_search_for_one_model():
    # ! TODO
    """This is to create pipeline test scenario one without params.

    Returns:
        [aml pipeline, pipeline and param_grid]: self explanatory
    """

    pipeline = Pipeline([
        ('disc1', EqualFrequencyDiscretiser()),
        ('disc2', EqualWidthDiscretiser()),
        ('model1', LinearRegression()),
        ('model2', RandomForestRegressor())
    ])

    param_grid = {
        'disc1__q': [5, 15],
        'model2__*': []
    }

    aml = AMLGridSearchCV(pipeline, param_grid)
    return aml, pipeline, param_grid


@pytest.fixture(scope="session")
def scenario_with_grid_search_for_whole_pipeline():
    # ! TODO
    """This is to create pipeline test scenario one without params.

    Returns:
        [aml pipeline, pipeline and param_grid]: self explanatory
    """

    pipeline = Pipeline([
        ('model1', LinearRegression()),
        ('model2', RandomForestRegressor())
    ])

    param_grid = {'*'}

    aml = AMLGridSearchCV(pipeline, param_grid)
    return aml, pipeline, param_grid
