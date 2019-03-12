import torch
import torch.nn as nn
import rationale_net.utils.learn as learn
import os
import pdb

NO_MODEL_ERR = "Model {} not in MODEL_REGISTRY! Available models are {}"

MODEL_REGISTRY = {}


def RegisterModel(model_name):
    """Registers a model."""

    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator


# Depending on arg, build model
def get_model(args, main_embeddings, secondary_embeddings):
    if args.snapshot is None:
        if args.model_form not in MODEL_REGISTRY:
            raise Exception(
                NO_MODEL_ERR.format(args.model_form, MODEL_REGISTRY.keys()))

        if args.model_form in MODEL_REGISTRY:
            if args.representation_type != 'both':
                model = MODEL_REGISTRY[args.model_form](args, main_embeddings)
            else:
                model = MODEL_REGISTRY[args.model_form](args, main_embeddings, secondary_embeddings)
        
    else :
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            gen_path = learn.get_gen_path(args.snapshot)
            if os.path.exists(gen_path):
                gen   = torch.load(gen_path)
            model = torch.load(args.snapshot)
        except :
            print("Sorry, This snapshot doesn't exist."); exit()

    if args.num_gpus > 1:
        model = nn.DataParallel(model,
                                device_ids=range(args.num_gpus))
    return model
