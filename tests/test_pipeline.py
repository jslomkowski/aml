from feature_engine.discretisation import EqualWidthDiscretiser
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def test_how_many_pipelines_are_created(scenario_one):
    aml, pipeline, param_grid = scenario_one
    final_pipes = aml._make_aml_combinations(pipeline, param_grid)

    assert len(final_pipes) == 4


def test_4th_step(scenario_one):
    aml, pipeline, param_grid = scenario_one
    final_pipes = aml._make_aml_combinations(pipeline, param_grid)

    steps = [('disc2', EqualWidthDiscretiser()),
             ('model2', RandomForestRegressor())]
    assert final_pipes[3].steps == steps


# final_pipes[3].steps[0][1] == steps[0][1]

# dir(final_pipes[3].steps[0][1])
# final_pipes[3].steps[0][1].__class__
