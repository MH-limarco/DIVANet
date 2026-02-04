# Model Training with DIVANet

## **Key Features of Train Mode**
>The following are some notable features of DIVANet's Train mode:
>+ **Hyperparameter Configuration**: The option to modify hyperparameters through YAML configuration files or python arguments.
>+ **Visualization and Monitoring**: Real-time tracking of training metrics and visualization of the learning process for better insights.
>
> Below is an example of how to resume an interrupted training using Python:

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
---
## DIVAN Settings

|               | Default | instruction                                                                                                                                |
|:--------------|:--------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| model_setting |         | Read YAML or pt, construct the framework used and initialize variables                                                                     |
| channels      | "RGB"   | Sets model input channels - support: RGB","RG","R","G","B","auto"...                                                                       |
| random_p      | 0.8     | Sets channel enable probability, only applies when channels == "auto"                                                                      |
| fix_mean      | False   | Enable averaging of clipping channels                                                                                                      |
| cut_channels  | False   | Enable Hard Clip Channel                                                                                                                   |
| amp           | True    | Enables Automatic Mixed Precision (AMP) training, reducing memory usage and possibly speeding up training with minimal impact on accuracy. |
| device        | "cuda"  | Specifies the computational device for training: auto-chose GPU(device='cuda'), a single GPU(device='cuda:0'),CPU(device='cpu).            |
| seed          | 0       | Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations.                           |
---
## Fit Settings

|                      | Default                              | instruction                                                                                                                                                     |
|:---------------------|:-------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset_path         |                                      | Path to the dataset file(e.g:'dataset')                                                                                                                         |
| epochs               |                                      | Sets training epochs(e.g:100)                                                                                                                                   |
| label_path           | ["train.txt", "val.txt", "test.txt"] | Path to the dataset label file                                                                                                                                  |
| size                 | 224                                  | Target image size for training. All images are resized to this dimension before being fed into the model. Affects model accuracy and computational complexity.  |
| batch_size           | 32                                   | Batch size                                                                                                                                                      |
| use_compile          | False                                | Enables model compile.Base on torch.compile                                                                                                                     |
| lr                   | 1e-3                                 | Initial learning rate                                                                                                                                           |
| weight_decay         | 0.01                                 | L2 regularization term, penalizing large weights to prevent overfitting.                                                                                        |
| warnup_step          | 0                                    | Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on. |
| warnup_start_factor  | 0.1                                  | Warnup learning rate as a fraction of the initial rate                                                                                                          |
| T_0                  | 3                                    | Number of iterations for the first restart                                                                                                                      |
| T_mult               | 2                                    | A factor increases 𝑇i after a restart                                                                                                                          |
| eta_min              | 0                                    | Minimum learning rate                                                                                                                                           |
| endstep_epochs       | 0                                    | Number of epochs for End-step(ReduceLROnPlateau)                                                                                                                |
| endstep_start_factor | 1                                    | End-step learning rate as a fraction of the initial rate                                                                                                        |
| endstep_patience     | 5                                    | Sets ReduceLROnPlateau number of allowed epochs with no improvement after which the learning rate will be reduced                                               |
| endstep_factor       | 0.1                                  | Sets ReduceLROnPlateau factor by which the learning rate will be reduced. new_lr = lr * factor                                                                  |
| early_stopping       | 15                                   | Sets early stopping limit                                                                                                                                       |
| label_smoothing      | 0.1                                  | Enables [label smoothing augmentation](https://arxiv.org/abs/1512.00567)                                                                                        |
| silence              | False                                | Enables logging silence mode                                                                                                                                    |
| cutmix_p             | 1                                    | Enables [CutMix augmentation](https://arxiv.org/abs/1905.04899v2)                                                                                               |
| pin_memory           | False                                | Enables pinned memory of dataloader                                                                                                                             |
| shuffle              | True                                 | Enables data reshuffled                                                                                                                                         |
| RAM                  | True                                 | Enables dataset pre-loading.If False, pre-loading must not enabled                                                                                              |
| ncols                | 90                                   | Sets width of the entire output message                                                                                                                         |
| RAM_lim              | 0.925                                | Sets the RAM usage limit when the dataset pre-loading ram check                                                                                                 |
| num_workers          | -1                                   | Sets the max workers of dataloader                                                                                                                              |