import sys


def validate_model_config(yml_model_config, model_default_config, arch):
    if arch == 'sk':

        if 'optimize' not in yml_model_config:
            print(
                'WARNING: No value for optimize aml_config.yml file. Using False')
            yml_model_config.update(
                {'optimize': ['False']})

        if 'metrics' not in yml_model_config:
            print(
                'WARNING: No metrics selected in aml_config.yml file. Using mean absolute error')
            yml_model_config.update(
                {'metrics': ['mean_absolute_error']})

        if 'models' not in yml_model_config:
            print('ERROR: model is required. Check aml_config.yml file')
            sys.exit(1)

            # TODO walidacja jeżeli użytkownik poda złą nazwe modelu metryki itp.
            #  np. xgbregressor a nie XGBRegressor

    elif arch == 'net':
        pass

    return yml_model_config


def validate_transformer_config(yml_transformer_config, trans_default_config):

    if not yml_transformer_config:
        yml_transformer_config.update(
            {'scalers': ['FunctionTransformer']})

    return yml_transformer_config


def validate_yml_config(yml_config):
    if any(v is None for v in yml_config.values()):
        print("TypeError: 'NoneType' object is not iterable. Check aml_config.yml for empty definitions eg. 'model:' without '- XGBRegressor'.")
        sys.exit(1)


def validate_config(cfg):

    if 'optimize' not in cfg:
        print(
            'WARNING: No value for optimize aml_config.yml file. Using False')
        cfg.update(
            {'optimize': 'False'})

    if 'metrics' not in cfg:
        print(
            'WARNING: No metrics selected in aml_config.yml file. Using mean absolute error')
        cfg.update(
            {'metrics': 'msle'})

    if 'models' not in cfg:
        print('ERROR: model is required. Check aml_config.yml file')
        sys.exit(1)

    return cfg
