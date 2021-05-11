
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.engine.input_spec import InputSpec


def _validate_steps(self):
    names, estimators = zip(*self.steps)

    # validate names
    self._validate_names(names)

    # validate estimators
    transformers = estimators[:-1]

    for t in transformers:
        if t is None or t == 'passthrough':
            continue


def Dense__init__(self,
                  units,
                  activation=None,
                  use_bias=True,
                  kernel_initializer='glorot_uniform',
                  bias_initializer='zeros',
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  activity_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None,
                  **kwargs):
    super(Dense, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs)

    # self.units = int(units) if not isinstance(units, int) else units
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.input_spec = InputSpec(min_ndim=2)
    self.supports_masking = True
