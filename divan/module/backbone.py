import torchvision, logging
import torch.nn as nn
import torch, timm

import contextlib, inspect, yaml, ast, re

from divan.module.kan_convs import *
from divan.module.block import *
from divan.module.conv import *
from divan.module.utils import *
from divan.module.attention import *
from divan.module.head import *

block_name = 'backbone'

__all__ = ["torch_model", "timm_model"]

def torch_model(model, usage_args=False, num_classes=50):
    weights = usage_args if usage_args is not False else None
    if type(model) == str:
        try:
            model = getattr(torchvision.models, model)

        except:
            logging.warning(f'{block_name}: model not exist')
            return None

    if not isinstance(usage_args, int):

        try:
            model = model(weights=weights)

        except:
            model = model(weights='DEFAULT')
            weights= 'DEFAULT'
            logging.warning(f'{block_name}: weights not exist')

        finally:
            logging.info(f'{block_name}: Loading model - {model.__name__}')
            logging.info(f'{block_name}: Loading weights - {weights}')

    else:
        model = inlayer_resize(model(False), usage_args)
    return model

def timm_model(model, usage_args=False, num_classes=50):
    weights = usage_args if usage_args is not False else None
    if type(model) == str:
        try:
            model = timm.create_model(model, pretrained=weights, num_classes=num_classes)

        except:
            try:
                model = timm.create_model(model, pretrained=False, num_classes=num_classes)
                weights = False
                logging.warning(f'{block_name}: weights not exist')
            except:
                logging.warning(f'{block_name}: model not exist')
                return None
        finally:
            logging.info(f'{block_name}: Loading model - {model.__class__.__name__}')
            logging.info(f'{block_name}: Loading weights - {weights}')
    else:
        model = timm.create_model(model, in_chans=usage_args, num_classes=num_classes)
    return model



if __name__ == '__main__':
    FORMAT = '[%(levelname)s] | %(asctime)s | %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt='%Y-%m-%d %H:%M')
    print(torch_model('resnet34'))



