import torch.nn as nn
import torch

import contextlib, logging, inspect, yaml, ast, re
import torchvision

from divan.module.kan_convs import *
from divan.module.block import *
from divan.module.conv import *
from divan.module.utils import *
from divan.module.attention import *
from divan.module.head import *

from divan.utils.config import *

try:
    import timm
except:
    pass

__all__ = ["vision_backbone", "timm_backbone"]

backbone_config = read_config(__file__)

def vision_backbone(model, weights=False, in_channels=3):
    if type(model) == str:
        try:
            model = getattr(torchvision.models, model)

        except:
            logging.warning(f'{backbone_config["block_name"]}: model not exist')
            return None

    try:
        model = model(weights=weights)

    except:
        model = model(weights='DEFAULT')
        weights= 'DEFAULT'
        logging.warning(f'{backbone_config["block_name"]}: weights not exist')

    finally:
        logging.info(f'{backbone_config["block_name"]}: Loading model - {model.__name__}')
        logging.info(f'{backbone_config["block_name"]}: Loading weights - {weights}')

    model = inlayer_resize(model, in_channels)
    return model

def timm_backbone(model, weights=False, in_channels=3):
    try:
        model = timm.create_model(model, pretrained=weights)

    except:
        try:
            model = timm.create_model(model, pretrained=False)
            weights = False
            logging.warning(f'{backbone_config["block_name"]}: weights not exist')
        except:
            logging.warning(f'{backbone_config["block_name"]}: model not exist')
            return None
    finally:
        logging.info(f'{backbone_config["block_name"]}: Loading model - {model.__class__.__name__}')
        logging.info(f'{backbone_config["block_name"]}: Loading weights - {weights}')
    model = inlayer_resize(model, in_channels)
    return model



if __name__ == '__main__':
    FORMAT = '[%(levelname)s] | %(asctime)s | %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt='%Y-%m-%d %H:%M')



