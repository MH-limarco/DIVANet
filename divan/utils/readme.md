# Setting with DIVANet
### You can modify your own setting yaml according to config_base.yaml
## **Key Features of Setting**
>The following are some notable features of DIVANet's Setting:
>+ **High scalability**: Easily change and create custom settings YAML
>
> You can create your own setting yaml file in divan/utils/config_file
> 
> After creating the yaml file, you only need to change the called yaml file in divan/utils/config.py
> 
>  Suppose I created config_test.yaml.  
>  Below is an example of how to change the called yaml file:


```python
#config.py
import ...

__all__ = ["apply_config", "read_config"]

use_version = "base"

...

```
---

## **config transform**
> The standard image augmentation pipeline of DIVANet is initialized in [cfg_transforms.py](config_file/)  
> Users can modify their own augmentation pipeline here.

