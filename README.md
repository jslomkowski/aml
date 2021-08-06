# AML

## This is WIP

experimental package for machine learning automated pipelines that iterates through transformers and estimators to return performance report with errors

## Install

```
pip install -e .
```
## Testing

Make sure Tox is installed
```
pip install tox
```
after that run:

```
tox -e test_pipeline
```

## Examples:
see github wiki page

## ToDo:
- [x] Tests
- [ ] Methods documentation
- [ ] CrossValidation
- [ ] New report
- [ ] Multiprocessing for NN (now it's only multithreading)
- [ ] Support for statsmodels - ??
- [ ] Time series split
- [x] Config dictionary for param_grid
- [x] Config dictionary for models / transformers
- [ ] Error handling
- [x] Examples