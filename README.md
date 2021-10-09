# AML

## This is WIP

experimental package for machine learning automated pipelines that iterates through transformers and estimators to return performance report with errors

## Install

```
pip install -e .
```

## Tests

```
tox -e test_pipeline
```

## Examples:
see github wiki page or documentation in /aml/docs/build/html/index.html

## ToDo:
- [x] Error handling (if pipe member breaks then continue)
- [ ] Execute one pipe from pipelines
- [ ] Be able to do only transformations (without modeling) (make tests)
- [x] Tests
- [x] Methods documentation
- [ ] CrossValidation
- [x] New report (make tests)
- [ ] Multiprocessing for NN (now it's only multithreading)
- [ ] Support for statsmodels - ??
- [ ] Time series split
- [x] Config dictionary for param_grid
- [x] Config dictionary for models / transformers
- [x] Examples
