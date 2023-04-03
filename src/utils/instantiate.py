from omegaconf import DictConfig
from functools import partial as partial_func
import importlib


def instantiate(config: DictConfig, partial: bool = False, reload: bool = False):
    target_key = "_target_"
    # partial_key = "_partial_"

    if target_key not in config:
        raise ValueError("Missing {} in config: {}".format(target_key, config))
    
    target_name = config.pop(target_key)

    # if config.pop(partial_key, False):
    if partial:
        return partial_func(get_obj_from_str(target_name, reload=reload), **config)
    else:
        return get_obj_from_str(target_name, reload=reload)(**config)

def get_obj_from_str(string: str, reload: bool = False):
    module_name, class_name = string.rsplit(".", 1)

    if reload:
        module = importlib.import_module(module_name)
        importlib.reload(module)

    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)