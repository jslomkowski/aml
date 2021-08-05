import shutil
import os
from sklearn.datasets import load_boston
import numpy as np
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
    check = [('disc2', EqualWidthDiscretiser()),
             ('model2', RandomForestRegressor())]
    assert str(final_pipes[3].steps) == str(check)


def test_default_models_template(scenario_with_default_models_template):
    aml, pipeline, param_grid = scenario_with_default_models_template
    final_pipes = aml._make_aml_combinations(pipeline, param_grid)
    check = [Pipeline(steps=[('disc1', EqualFrequencyDiscretiser()),
                             ('model1', LinearRegression())]),
             Pipeline(steps=[('disc1', EqualFrequencyDiscretiser()),
                             ('model2', RandomForestRegressor())]),
             Pipeline(steps=[('disc2', EqualWidthDiscretiser()),
                             ('model1', LinearRegression())]),
             Pipeline(steps=[('disc2', EqualWidthDiscretiser()),
                             ('model2', RandomForestRegressor())])]
    assert str(final_pipes) == str(check)


def test_custom_models_template(scenario_with_custom_models_template):
    aml, pipeline, param_grid = scenario_with_custom_models_template
    final_pipes = aml._make_aml_combinations(pipeline, param_grid)
    check = [Pipeline(steps=[('disc1', EqualFrequencyDiscretiser()),
                             ('model1', LinearRegression())]),
             Pipeline(steps=[('disc1', EqualFrequencyDiscretiser()),
                             ('model2', RandomForestRegressor())]),
             Pipeline(steps=[('disc2', EqualWidthDiscretiser()),
                             ('model1', LinearRegression())]),
             Pipeline(steps=[('disc2', EqualWidthDiscretiser()),
                             ('model2', RandomForestRegressor())])]
    assert str(final_pipes) == str(check)


def test_grid_search_for_one_model(scenario_with_grid_search_for_one_model):
    aml, pipeline, param_grid = scenario_with_grid_search_for_one_model
    pipeline_steps_list = aml._models_template_check(pipeline)
    grid = aml._check_def_config(pipeline_steps_list, param_grid)
    check = {
        'disc1__q': [5, 15],
        'model2__n_estimators': [100],
        'model2__max_features': np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                          0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                                          0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                                          0.95, 1.]),
        'model2__min_samples_split': range(2, 21),
        'model2__min_samples_leaf': range(1, 21),
        'model2__bootstrap': [True, False]}
    assert str(grid) == str(check)


def test_grid_search_for_whole_pipeline(scenario_with_grid_search_for_whole_pipeline):
    aml, pipeline, param_grid = scenario_with_grid_search_for_whole_pipeline
    pipeline_steps_list = aml._models_template_check(pipeline)
    grid = aml._check_def_config(pipeline_steps_list, param_grid)
    check = {
        'model1__normalize': [True, False],
        'model1__positive': [True, False],
        'model2__n_estimators': [100],
        'model2__max_features': np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                          0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                                          0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                                          0.95, 1.]),
        'model2__min_samples_split': range(2, 21),
        'model2__min_samples_leaf': range(1, 21),
        'model2__bootstrap': [True, False]}
    assert str(grid) == str(check)


def test_reports(scenario_without_params):
    X, y = load_boston(return_X_y=True)
    aml, pipeline, param_grid = scenario_without_params
    aml.fit(X, y, save_report=True, report_format='csv')
    aml.fit(X, y, save_report=True, report_format='xlsx')
    aml.fit(X, y, save_report=True, report_format='abc')
    check = os.listdir('aml_reports')[0][-3:] + \
        os.listdir('aml_reports')[1][-4:]
    shutil.rmtree('aml_reports')
    assert check == 'csvxlsx'
