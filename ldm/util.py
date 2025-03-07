import importlib

def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise ValueError(f"config must have a 'target' key: {config}")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

# 从config里加载库与对应模型
def get_obj_from_str(string, reload=False):
    module,cls = string.rsplit(".",1)
    if reload:
        importlib.reload(importlib.import_module(module))
    return getattr(importlib.import_module(module, package=None), cls)
        