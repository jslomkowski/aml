from test_scenarios import scenario_one
from feature_engine.discretisation import EqualWidthDiscretiser
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

aml, pipeline, param_grid = scenario_one()


def test_how_many_pipelines_are_created():
    final_pipes = aml._make_aml_combinations(pipeline, param_grid)
    assert len(final_pipes) == 4


def test_4th_step():
    final_pipes = aml._make_aml_combinations(pipeline, param_grid)
    steps = [('disc2', EqualWidthDiscretiser()),
             ('model2', RandomForestRegressor())]
    assert final_pipes[3].steps == steps


# final_pipes[3].steps[0][1] == steps[0][1]

# dir(final_pipes[3].steps[0][1])
# final_pipes[3].steps[0][1].__class__
