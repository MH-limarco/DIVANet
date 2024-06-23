# DIVANet: Dynamic Interactive Visual Architecture Network




This is presented as my second practical exercise, and feedback or issues are welcome on GitHub.

* [**Documentation**](#documentation)
* [**Getting started**](#getting-started)
* [**Introduction to structure**](#introduction-to-structure)
* [**Customize your process**](#customize-your-process)


## <div align="center">👟Open modular framework👟</div>

## <div align="center">📔Documentation📔</div>

See below for a quickstart and usage example

<details open>
<summary>Preliminaries</summary>

### Environment
This project is based on a [**Python>=3.11**](https://www.python.org/) and cuda>=11.8 environment with Windows 11 & Ubuntu 20.04.

### Package install
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install pandas
pip install numpy
pip install psutil

#Optional
pip install timm
```

For alternative installation methods including [Conda](https://anaconda.org/conda-forge/pandas).

</details>

## <div align="center">👐Getting started👐</div>

<details open>
<summary>Usage</summary>


### Python - Training
```python
from divan import DIVAN

if __name__ == '__main__':
    model = DIVAN('resnet34.yaml')
    model.fit('dataset', 
              epochs=100,
              warnup_step=5,
              endstep_epochs=20,
              endstep_patience=3,
              endstep_factor=0.5,
              batch_size=128,
              cutmix_p=0,
              label_smoothing=0.0,
              lr=0.0005,
              early_stopping=48,
              RAM=True)
```

### Python - Continuing training
```python
from divan import DIVAN

if __name__ == '__main__':
    model = DIVAN('divan_result/train/resnet34-1/weight/last.pt')
    model.fit('dataset', 
              epochs=100,
              warnup_step=5,
              endstep_epochs=20,
              endstep_patience=3,
              endstep_factor=0.5,
              batch_size=128,
              cutmix_p=0,
              label_smoothing=0.0,
              lr=0.0005,
              early_stopping=48,
              RAM=True)
```
</details>


## <div align="center">📋API Introduction📋</div>

<details open>

### [DIVAN](divan/readme.md)

### [Configuration](cfg/readme.md)

### [Setting](divan/utils/config_file/readme.md)

</details>

## <div align="center">⚒️TODO⚒️</div>
- [ ] **wandb support**
- [ ] **Distributed Data Parallel**
- [x] **Continuing training**