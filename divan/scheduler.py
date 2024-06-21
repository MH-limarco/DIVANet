import torch

__all__ = ["FactorReduceLROnPlateau"]

class FactorReduceLROnPlateau:
    def __init__(self, optimizer, start_factor=1, mode='min', patience=10, threshold=0.0001, factor=0.1):
        self.optimizer = optimizer
        self.optimizer_lr = optimizer.param_groups[0]['lr']
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

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

if __name__ == "__main__":
    lr = 1
    warnup=4
    model = torch.nn.Sequential(torch.nn.Linear(1, 10))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_start = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                           T_0=10,
                                                                           T_mult=2,
                                                                           eta_min=0,
                                                                           last_epoch=-1)
    scheduler_warn = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=warnup)

    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler_warn, scheduler_start], milestones=[warnup])
    sc = FactorReduceLROnPlateau(optimizer, patience=3, start_factor=0.5)
    for epoch in range(30+warnup-2):
        scheduler.step()
        if epoch == 20:
            mew = optimizer.state_dict()
        print(epoch, optimizer.param_groups[0]['lr'])

    optimizer.load_state_dict(mew)
    print('hello', round(optimizer.param_groups[0]['lr'], 5))
    sc.init_lr()
    sc.step(0.1)
    print('end', round(optimizer.param_groups[0]['lr'], 5))
    for epoch in range(15):
        sc.step(0.11)
        print(optimizer.param_groups[0]['lr'])
