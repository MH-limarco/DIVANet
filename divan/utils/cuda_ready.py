import logging

import torch.nn as nn
import torch
import subprocess

memory_used_command = "nvidia-smi --query-gpu=memory.used --format=csv"
block_name = 'CUDA_setting'
def chose_cuda(device):
    if device == 'cuda' and torch.cuda.device_count() > 1:
        cuda_ram = subprocess.run(memory_used_command.split(' '),encoding='utf-8',
                             stdout=subprocess.PIPE, stdin=subprocess.PIPE).stdout.replace('MiB', '')
        cuda_ram = [int(i) for i in cuda_ram.split('\n')[1:] if len(i) > 0]
        device = f'{device}:{min(range(len(cuda_ram)), key=cuda_ram.__getitem__)}'
        logging.debug(f'{block_name}: Auto choose - {device}')
        return device

    else:
        return device

if __name__ == '__main__':
    print(chose_cuda('cuda'))