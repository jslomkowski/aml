import itertools


def net_settings():

    loss = [
        'mean_squared_error',
        # 'mean_absolute_error',
        # 'mean_absolute_percentage_error',
        # 'mean_squared_logarithmic_error',
        # 'cosine_similarity',
        # 'logcosh',
    ]

    metrics = [
        # 'mean_squared_error',
        # 'mean_absolute_error',
        'mean_absolute_percentage_error',
        # 'mean_squared_logarithmic_error',
        # 'cosine_similarity',
        # 'logcosh',
    ]

    optimizer = [
        'adam',
        # 'sgd',
    ]

    batch_size = [
        # 1,
        # 2,
        # 4,
        # 8,
        # 16,
        # 32,
        # 64,
        # 128,
        256
    ]

    epochs = [
        # 1,
        # 2,
        # 4,
        # 8,
        # 16,
        # 32,
        # 64,
        # 128,
        256
    ]

    neurons1 = [
        # 1,
        # 4,
        # 16,
        64,
        # 256
    ]

    neurons2 = [
        # 1,
        # 4,
        # 16,
        64,
        # 256
    ]

    neurons3 = [
        # 1,
        # 4,
        # 16,
        64,
        # 256
    ]

    layers = [
        1,
        # 2,
        # 3
    ]

    if not metrics:
        metrics = ['delete']

    # combine all variables into single dict
    settings = {'loss': loss, 'metrics': metrics, 'optimizer': optimizer,
                'batch_size': batch_size, 'epochs': epochs, 'layers': layers,
                'neurons1': neurons1, 'neurons2': neurons2, 'neurons3': neurons3}

    # unpack that dict
    settings = list((dict(zip(settings, x))
                     for x in itertools.product(*settings.values())))

    # if something is named 'delete' then convert it to None
    for s in settings:
        for key, value in s.items():
            if value == 'delete':
                s[key] = None
        # if user specified less than 3 neurons remove excess
        if s['layers'] == 1:
            s['neurons2'] = None
            s['neurons3'] = None
        if s['layers'] == 2:
            s['neurons3'] = None

    # remove duplicated dictionary values
    settings = [dict(t) for t in {tuple(d.items()) for d in settings}]
    return settings
