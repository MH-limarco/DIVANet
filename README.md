# 👻DIVANet: Dynamic Interactive Visual Architecture Network

>## 🎯**Introduction**
> This project is an extension of the Institute of 1122 - NCKU Data Science's Deep Learning HW2 requirement. It aims to introduce an open framework to the standard training process, enhancing its scalability and usability. Users can easily integrate updated models or modules into the framework and combine them seamlessly.
> 
>Additionally, a straightforward automatic equipment allocation feature has been incorporated to maximize the computational performance of the computing center.
>
>## 📑Contents:
>- #### [Documentation](#documentation)
> 
>- #### [Getting started](#getting-started-1)
> 
>- #### [API Introductione](#API-Introduction)


## <div align="center">📔Documentation📔</div>

#### See below for a quickstart

<details open>
<summary>🙈Preliminaries</summary>

### 🚩Environment
>This project is based on a [**Python>=3.11**](https://www.python.org/) and [**CUDA Toolkit>=11.8**](https://developer.nvidia.com/cuda-toolkit) environment with Windows 11 & Ubuntu 20.04.

### 💻Package install
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
>⚠️ For alternative installation methods including [Conda](https://anaconda.org/conda-forge/pandas).
</details>

## <div align="center">👐Getting started👐</div>
#### See below for a usage example
<details open>
<summary>🏃Usage</summary>

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
### Python - Training(Custom-Model)
>  ⚠️In custom models, you can append "-{scale name}" to the YAML file name to specify model parameter scaling.  
>
>  **support scale: "n", "s", "m", "l", "x"**  
>  For example, use "C2f-DCNv4-x.yaml".  

```python
from divan import DIVAN

if __name__ == '__main__':
    model = DIVAN('C2f-DCNv4-x.yaml')
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

>#### ⚠️After the training is completed, the training process information and parameters will be recorded in:
>divan_result/train/{yaml_name}-{id} (e.g: divan_result/train/resnet34-1)

</details>


## <div align="center">📋API Introduction📋</div>
#### Each part is described in detail as follows:
<details open>
<summary>📑Menu</summary>

+ #### [DIVAN](divan)

+ #### [Configuration](cfg)

+ #### [Setting](divan/utils)

</details>

## 📧Concat
> #### email: nn6125010@gs.ncku.edu.tw

## 💡Acknowledgement
>DIVANet is built with reference to the code of the following projects:
>* #### [Ultralytics](https://github.com/ultralytics/ultralytics?tab=readme-ov-file)
>* #### [DCNv4](https://github.com/OpenGVLab/DCNv4)
>* #### [torch-conv-kan](https://github.com/IvanDrokin/torch-conv-kan)
>Thanks for their awesome work!

## 📃TODO
✅ **Continuing training**  
📝 **wandb support**  
📝 **Unified log management**  
📝 **Distributed Data Parallel**  
📝 **Separate testing function**  

---
This is presented as my second practical exercise, and feedback or issues are welcome on GitHub.
