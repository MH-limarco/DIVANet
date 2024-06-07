# DIVANet: Dynamic Interactive Visual Architecture Network


TODO:

This is presented as my second practical exercise, and feedback or issues are welcome on GitHub.

* [**Documentation**](#documentation)
* [**Getting started**](#getting-started)
* [**Introduction to structure**](#introduction-to-structure)
* [**Customize your process**](#customize-your-process)


## <div align="center">👟Open modular framework👟</div>
<p ="center">
  <img src="assets/TODO" alt="erf" width="100%">
</p>


## <div align="center">📔Documentation📔</div>

See below for a quickstart and usage example

<details open>
<summary>Preliminaries</summary>

### Environment
This project is based on a [**Python>=3.11**](https://www.python.org/) environment with Windows 11 & Ubuntu 20.04.

### Package install
```bash
TODO
```

For alternative installation methods including [Conda](https://anaconda.org/conda-forge/pandas), and Git.

</details>

## <div align="center">👐Getting started👐</div>

<details open>
<summary>Usage</summary>


### Python - build model with yaml

TODO

```python
from divan import DIVAN

if __name__ == '__main__':
    model = DIVAN()
    model.fit("C2f.yaml",
              "dataset",
              epochs=10,
              )
```

### Python - read model with torchvision

TODO

```python
from divan import DIVAN_torch

if __name__ == '__main__':
    model = DIVAN_torch()
    model.fit("resnet34",
              "dataset",
              epochs=10,
              )
```

</details>

## <div align="center">🔖Introduction to structure🔖</div>

|                |                                                                       instruction                                                                       |
|:--------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      TODO      |                                            TODO                                          |
| TODO |  TODO  |
|  TODO   |          TODO                                       |


## <div align="center">⚒️Architecture example⚒️</div>
<p = "center">
  <img src="assets/TODO" alt="erf" width="100%">
</p>

#### Here is the architecture example for `cfg/C2f.yaml`.
You can build your own structure.yaml according to your requirements

## <div align="center">📋Customize your process📋</div>
<details open>
<summary>setting.yaml</summary>

### Create a setting yaml.

You need to customize your configuration file so that the script knows where your architecture files are, where to input and output, and the settings for execution.

|                        |                         example                         | definition                                                                                  |
|:----------------------:|:-------------------------------------------------------:|:--------------------------------------------------------------------------------------------|
    |          TODO          |                        C2f.yaml                         | Architecture file PATH                                                                      |
|          TODO          |                            TODO                             | TODO                                                                                        |
|          TODO          |                           TODO                            | TODO                                                                                        |
|          TODO          |                           TODO                            | TODO                                                                                        |
|      TODO       |                            TODO                           | TODO                                                   |

</details>


<details open>
<summary>structure.yaml</summary>
           - [-2, ot_df_concat, []]...                | Output-step blueprint settings    |

#### blueprint-format

```angular2html
blueprint-format
- [from_idx, n, module, args] #idx
```

#### blueprint-idx

```angular2html
from_idx:
- -1: pass value
- other: module idx
```

For existing modules, you can learn more by visiting the [[**Doc**]](assets%2FREADME.md).

</details>
