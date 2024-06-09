import torch
import yaml

from divan.utils.utils import *

class Config:
    def __init__(self, version='en'):
        #with open(f"divan/utils/config_file/config_{version}.yaml",
        with open(f"config_file/config_{version}.yaml",
                  mode='r', encoding="utf-8"
                  ) as stream:
            _config = yaml.safe_load(stream)

        set_arg(self)
        set_kargs(self, _config)

        print(locals())
        print(__file__)



if __name__ == "__main__":
    Config()

