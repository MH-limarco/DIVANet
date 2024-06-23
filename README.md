# DIVANet: Dynamic Interactive Visual Architecture Network

>## **Introduction**
> This project is based on the Institute of Data Science 1122 Deep Learning HW2 requirement expansion.
> I try to add an open framework to the normal training process, so that the entire training framework can be highly scalable and easy to use.
> Users can easily add updated models or modules here and combine them with each other on this basis.
> 
> On the other hand, I added a simple automatic equipment allocation function to make full use of the computing center's computing performance as much as possible.
> 
>## Contents:
>#### [**Documentation**](#documentation)
> 
>#### [**Getting started**](#Getting-started)
> 
>#### [**API Introductione**](#API-Introduction)


## <div align="center">📔Documentation📔</div>

#### See below for a quickstart and usage example

<details open>
<summary>Preliminaries</summary>

### Environment
>This project is based on a [**Python>=3.11**](https://www.python.org/) and cuda>=11.8 environment with Windows 11 & Ubuntu 20.04.

### Package install
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install pandas
pip install numpy
pip install psutil

#Optional timm
pip install timm

#Optional DCNv4
cd extra_modules/DCNv4_op
python setup.py build install
```
>For alternative installation methods including [Conda](https://anaconda.org/conda-forge/pandas).
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
>#### After the training is completed, the training process information and parameters will be recorded in:
>divan_result/train/{yaml_name}-{id}

</details>


## <div align="center">📋API Introduction📋</div>

<details open>

>### [DIVAN](divan/readme.md)
>
>### [Configuration](cfg/readme.md)
>
>### [Setting](divan/utils/config_file/readme.md)

</details>

## TODO
>- [ ] **wandb support**
>- [ ] **Distributed Data Parallel**
>- [ ] **Separate testing function**
>- [x] **Continuing training**

## Concat
>nn6125010@gs.ncku.edu.tw

## Acknowledgement
>DIVANet is built with reference to the code of the following projects
>* #### [Ultralytics](https://github.com/ultralytics/ultralytics?tab=readme-ov-file)
>* #### [DCNv4](https://github.com/OpenGVLab/DCNv4)
>Thanks for their awesome work!

---
This is presented as my second practical exercise, and feedback or issues are welcome on GitHub.
