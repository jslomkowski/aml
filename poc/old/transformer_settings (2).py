import itertools

from feature_engine.discretisation import *
from feature_engine.encoding import *
from feature_engine.imputation import *
from feature_engine.outliers import *
from feature_engine.selection import *
from feature_engine.transformation import *
from sklearn.preprocessing import *


def transformer_settings():

    imputers = [
        # MeanMedianImputer(),
        # ArbitraryNumberImputer(),
        # EndTailImputer(),
        # CategoricalImputer(),
        # RandomSampleImputer()
    ]

    encoders = [
        # OneHotEncoder(),
        CountFrequencyEncoder(),
        OrdinalEncoder(),
        MeanEncoder(),
        DecisionTreeEncoder(),
        # RareLabelEncoder()
    ]

    transformers = [
        # LogTransformer(),
        # ReciprocalTransformer(),
        # # PowerTransformer(),
        # BoxCoxTransformer(),
        # YeoJohnsonTransformer()
    ]

    discretisers = [
        # EqualFrequencyDiscretiser(),
        # EqualWidthDiscretiser(),
        # # ArbitraryDiscretiser(),
        # DecisionTreeDiscretiser()
    ]

    outlier_handlers = [
        # Winsorizer(),
        # # ArbitraryOutlierCapper(),
        # OutlierTrimmer()
    ]

    selectors = [

    ]

    scalers = [
        # FunctionTransformer(),
        # StandardScaler(),
        # Normalizer(),
    ]

    # combine all variables into single dict
    settings = {'imputers': imputers, 'encoders': encoders,
                'transformers': transformers, 'discretisers': discretisers,
                'outlier_handlers': outlier_handlers, 'selectors': selectors,
                'scalers': scalers
                }

    # remove all empty keys
    settings = {k: v for k, v in settings.items() if v != []}

    # if no transformers are selected then inject FunctionTransformer w identity function
    if not settings:
        settings['scalers'] = [FunctionTransformer()]

    # unpack that dict
    settings = list((dict(zip(settings, x))
                     for x in itertools.product(*settings.values())))

    # remove duplicated dictionary values
    settings = [dict(t) for t in {tuple(d.items()) for d in settings}]
    return settings
