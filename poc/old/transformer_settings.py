import itertools

from feature_engine.discretisation import *
from feature_engine.encoding import *
from feature_engine.imputation import *
from feature_engine.outliers import *
from feature_engine.selection import *
from feature_engine.transformation import *
from sklearn.preprocessing import *


def transformer_default_config():
    default_config = {

        # imputers
        'MeanMedianImputer': MeanMedianImputer(),
        'ArbitraryNumberImputer': ArbitraryNumberImputer(),
        'EndTailImputer': EndTailImputer(),
        'CategoricalImputer': CategoricalImputer(),
        'RandomSampleImputer': RandomSampleImputer(),

        # encoders
        'OneHotEncoder': OneHotEncoder(),
        'CountFrequencyEncoder': CountFrequencyEncoder(),
        'OrdinalEncoder': OrdinalEncoder(),
        'MeanEncoder': MeanEncoder(),
        'DecisionTreeEncoder': DecisionTreeEncoder(),
        'RareLabelEncoder': RareLabelEncoder(),

        # transformers
        'LogTransformer': LogTransformer(),
        'ReciprocalTransformer': ReciprocalTransformer(),
        'BoxCoxTransformer': BoxCoxTransformer(),
        'YeoJohnsonTransformer': YeoJohnsonTransformer(),

        # discretisers
        'EqualFrequencyDiscretiser': EqualFrequencyDiscretiser(),
        'EqualWidthDiscretiser': EqualWidthDiscretiser(),
        'DecisionTreeDiscretiser': DecisionTreeDiscretiser(),

        # # outlier_handlers
        # 'Winsorizer': Winsorizer(),
        # 'OutlierTrimmer': OutlierTrimmer(),

        # scalers
        'FunctionTransformer': FunctionTransformer(),
        'StandardScaler': StandardScaler(),
        'Normalizer': Normalizer(),
    }

    return default_config
