# Model Configuration with DIVANet

## **Key Features of Model Configuration**
>The following are some notable features of DIVANet's Model Configuration:
>+ **High usability**: Create custom models by YAML format
>+ **High scalability**: Easily call different models and modules and support complex computing architectures

>Supports a wide range of models and modules, and users can easily integrate their own models or modules as needed, combining them seamlessly.
> 
> Below is an example of how to resume an interrupted training using yaml:

### YAML - Torchvision-Resnet34
```yaml
# Parameters
nc: 1000 # number of classes
scales:
  # [depth, width, max_channels]
  n: [1, 1, 1024]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, vision_backbone, ['resnet34']]
```
---
### YAML - timm-HGnetv2_b0
#### Need to install timm
```yaml
# Parameters
nc: 1000 # number of classes
scales:
  # [depth, width, max_channels]
  n: [1, 1, 1024]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, timm_backbone, ['hgnetv2_b0']]
```
---
### YAML - Simple Branching CNN
```yaml
# Parameters
nc: 1000 # number of classes
activation: Mish
scales:
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 1024]
  l: [1.00, 1.00, 1024]
  x: [1.00, 1.25, 1024]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, nn.Identity, []] #0

  - [-1, 1, Conv, [64, 3, 2]]    # 1
  - [-1, 1, Conv, [256, 3, 2]]   # 2

  - [0, 1, Conv, [64, 3, 2]]     # 3
  - [-1, 1, Conv, [256, 3, 2]]   # 4

  - [0, 1, Conv, [64, 3, 2 ]]     # 5
  - [-1, 1, Conv, [256, 3, 2 ]]   # 6

  - [[2, 4, 6], 1, Concat,[]]     # 7
  - [-1, 1, Conv, [512, 3, 2 ]]   # 8

head:
  - [-1, 1, Classify, [nc]] # Classify
```
___
### YAML - C2f_KANConv with CBAM
```yaml
# Parameters
nc: 1000 # number of classes
activation: Mish
scales:
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 1024]
  l: [1.00, 1.00, 1024]
  x: [1.00, 1.25, 1024]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2     #1
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4   #1
  - [-1, 1, C2f_FastKANConv, [128, True]]   # 2
  - [-1, 1, CBAM, []]

head:
  - [-1, 1, Classify, [nc]] # Classify    #1
```
### YAML - C2f with DCNv4
```yaml
# Parameters
nc: 1000 # number of classes
activation: Mish
scales:
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 1024]
  l: [1.00, 1.00, 1024]
  x: [1.00, 1.25, 1024]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]         # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]        # 1-P2/4
  - [-1, 1, C2f_DCNv4, [128, True]]   # 2

head:
  - [-1, 1, Classify, [nc]] # Classify   
```

---