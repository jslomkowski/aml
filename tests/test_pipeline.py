from aml.models_template import my_models_list
from feature_engine.discretisation import (EqualFrequencyDiscretiser,
                                           EqualWidthDiscretiser)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def test_how_many_pipelines_are_created(scenario_without_params):
    aml, pipeline, param_grid = scenario_without_params
    final_pipes = aml._make_aml_combinations(pipeline, param_grid)

    assert len(final_pipes) == 4


def test_4th_step_in_scenario_without_params(scenario_without_params):

    aml, pipeline, param_grid = scenario_without_params
    final_pipes = aml._make_aml_combinations(pipeline, param_grid)

    steps = [('disc2', EqualWidthDiscretiser()),
             ('model2', RandomForestRegressor())]

    assert str(final_pipes[3].steps) == str(steps)


def test_import_from_models_template(scenario_with_import):

    aml, pipeline, param_grid = scenario_with_import
    final_pipes = aml._make_aml_combinations(pipeline, param_grid)

    pipelines = [Pipeline(steps=[('disc1', EqualFrequencyDiscretiser()),
                                 ('model1', LinearRegression())]),
                 Pipeline(steps=[('disc1', EqualFrequencyDiscretiser()),
                                 ('model2', RandomForestRegressor())]),
                 Pipeline(steps=[('disc2', EqualWidthDiscretiser()),
                                 ('model1', LinearRegression())]),
                 Pipeline(steps=[('disc2', EqualWidthDiscretiser()),
                                 ('model2', RandomForestRegressor())])]

    assert str(final_pipes) == str(pipelines)
