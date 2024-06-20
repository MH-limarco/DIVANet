import torch

__all__ = ["FactorReduceLROnPlateau"]

class FactorReduceLROnPlateau:
    def __init__(self, optimizer, start_factor=0.8, mode='min', patience=10, threshold=0.0001, factor=0.1):
        self.optimizer = optimizer
        self.start_factor = start_factor
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold)
        self.init_lr()

    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.start_factor

    def step(self, metrics):
        self.scheduler.step(metrics)

    def state_dict(self):
        return self.scheduler.state_dict()

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]