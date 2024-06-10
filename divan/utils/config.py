import torch
import yaml, os

from divan.utils.utils import *
from divan.utils.config_file import *

__all__ = ["apply_config"]

use_version = 'en'

class project_config:
    def __init__(self, version='en'):
        with open(f"divan/utils/config_file/config_{version}.yaml",
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

Config = project_config(version=use_version)
