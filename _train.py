import numpy as np

from divan import *
from divan.utils.dataset import DIVANetDataset
from divan.check.check_file import check_file
from tqdm import tqdm
import time
from divan.parse_task import Divanet_model

if __name__ == "__main__":
    FORMAT = '%(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        format=FORMAT)
    yaml_path = "resnet34"
    def _read_yaml(yaml_path):
        with open(f'cfg/{yaml_path}.yaml', 'r', encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    device = 'cuda:0'
    model = Divanet_model(_read_yaml(yaml_path), 3, device).to(device)

    model2 = torch_model('resnet34', 3, num_classes=50).to(device)

    imgs = torch.randn((64, 3, 224, 224))
    for i in tqdm(range(1000)):
        _input = imgs.to(device)
        with torch.autocast(device_type='cuda'):
            model(_input)

    for i in tqdm(range(1000)):
        _input = imgs.to(device)
        with torch.autocast(device_type='cuda'):
            model2(_input)

