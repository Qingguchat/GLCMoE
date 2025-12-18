
import yaml

class Config:

    def __init__(self, dictionary: dict):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return f"Config({self.__dict__})"

def load_config(path: str) -> Config:

    with open(path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)

    cfg = Config(cfg_dict)

    for section in ['data', 'train', 'inference', 'model', 'local_moe', 'global_moe']:
        if not hasattr(cfg, section):
            raise KeyError(f"config file must contain '{section}' section")

    return cfg
