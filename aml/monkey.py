
def _validate_steps(self):
    names, estimators = zip(*self.steps)

    # validate names
    self._validate_names(names)

    # validate estimators
    transformers = estimators[:-1]

    for t in transformers:
        if t is None or t == 'passthrough':
            continue
