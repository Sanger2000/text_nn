
NO_MODEL_ERR = "Model {} not in MODEL_REGISTRY! Available models are {}"

MODEL_REGISTRY = {}


def RegisterModel(model_name):
    """Registers a model."""

    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator


# Depending on arg, build model
def get_model(args, word_to_indx, char_to_indx, truncate_train=False):
    if args.model_form not in MODEL_REGISTRY:
        raise Exception(
            NO_MODEL_ERR.format(args.model_form, MODEL_REGISTRY.keys()))

    if args.model_form in MODEL_REGISTRY:
        train = MODEL_REGISTRY[args.model_form](args, 'train')
        dev = MODEL_REGISTRY[args.model_form](args, word_to_indx, char_to_indx, 'dev')
        test = MODEL_REGISTRY[args.model_form](args, word_to_indx, char_to_indx, 'test')

    return train, dev, test
