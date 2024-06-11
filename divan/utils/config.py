import torch
import yaml, os

from divan.utils.utils import *
from divan.utils.config_file import *

__all__ = ["apply_config", "read_config"]

use_version = "en"

class project_config:
    def __init__(self, version='en'):
        file_path = os.sep.join(__file__.split(os.sep)[:-1])
        with open(f"{file_path}/config_file/config_{version}.yaml",
                  mode='r', encoding="utf-8") as stream:
            _config = yaml.safe_load(stream)
        apply_args(self)
        apply_kwargs(self, _config)
        self.dataset['transforms'] = Transforms

def apply_config(_class, file):
    file_index = file.split(os.sep)[-1].split('.')[0]
    _config = getattr(Config, file_index)
    apply_kwargs(_class, _config)
    apply_kwargs(_class, getattr(Config, 'base'))

def read_config(file):
    file_index = file.split(os.sep)[-1].split('.')[0]
    return getattr(Config, file_index)

Config = project_config(version=use_version)
