import contextlib, inspect, ast, logging
import torch.nn as nn

from divan.module.utils import *
from divan.module.attention import *
from divan.module.backbone import *
from divan.module.block import *
from divan.module.conv import *
from divan.module.head import *
from divan.module.kan_convs import *

from divan.utils.config import *
from divan.utils.utils import *

class Divanet_model(nn.Module):
    def __init__(self, _dict, _input_channels):
        super().__init__()
        apply_args(self)
        apply_config(self, __file__)
        self.sequential, self.save_idx, self.fc_resize = self._parse_step(_dict, _input_channels)

    def forward(self, x):
        out = {-1: x}
        for module in self.sequential:
            save_i = [-1] + [i for i in [module.i] if i in self.save_idx]
            _input = [out[k] for k in module.f] if len(module.f) > 1 else out[module.f[0]]
            _output = module(_input)
            out.update(out.fromkeys(save_i, _output))
        return out[-1]

    def _parse_step(self, _dict, _input_channels, input_pool='Pool_Conv'): # model_dict, input_channels(3)
        max_channels = float("inf")
        fc_resize = True
        num_class, act, scales, c1_pool = (_dict.get(x) for x in ("nc", "activation", "scales","c1_pool"))

        if scales:
            scale = _dict.get("scale")
            if not scale:
                scale = tuple(scales.keys())[0]
            logging.info(f"{self.block_name}: Assuming scale='{scale}'")
                # LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            Conv.default_act = Activations(act)(inplace=True)
            #print(f"{('activation:')} {Conv.default_act}")  # print

        #if verbose:
            #LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

        ch = [_input_channels]
        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

        if _input_channels == 'auto':
            ch = [1]
            input_pool = globals()[input_pool]
            m_seq = input_pool(*[16, 3, Conv, act, ['Avg', 'Max']])

            t = str(input_pool)[8:-2].replace("__main__.", "")
            input_pool.np = sum(x.numel() for x in m_seq.parameters())  # number params
            m_seq.i, m_seq.f, m_seq.type = -1, [-1], t
            layers.append(m_seq)

        full_dict = _dict["backbone"]
        if _dict.get("head"):
            full_dict += _dict["head"]
        else:
            print('no head')

        for i, (f, n, m, args) in enumerate(full_dict):  # from, number, module, args
            assert max([f] if isinstance(f, int) else f) < i
            m = m.split('_') if m not in ['vision_backbone', 'timm_backbone'] else [m]
            if len(m) == 2:
                if 'KA' in m[1] and 'NConv' in m[1]:
                    m, _conv = m[0] + '_KAN', m[1]
                elif m[0] + m[1] in globals():
                    m, _conv = m[0] + m[1], None
                else:
                    m, _conv = m[0], m[1]

            elif len(m) == 1:
                m, _conv = m[0], None
            else:
                raise ValueError(f'invalid model: {m}')

            m = getattr(nn, m[3:]) if m.startswith('nn.') else globals()[m]
            _conv = globals()[_conv] if _conv else Conv

            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

            if m not in [nn.BatchNorm2d, nn.Identity,
                         Concat,
                         vision_backbone, timm_backbone,
                         ChannelAttention, SpatialAttention, CBAM
                         ]:
                c1, c2 = ch[f], args[0]

                if c2 != 'nc':  # if c2 not equal to number of classes (i.e. for Classify() output)
                    c2 = make_divisible(min(c2, max_channels) * width, 8)

                else:
                    c2 = num_class

                args = [c1, c2, *args[1:]]
                if m in {C1, C2, C2f, C3, C2f_KAN, C3_KAN}:
                    _arg = inspect.getfullargspec(m)[0]
                    args.insert(_arg.index('n') - 1 if 'self' in _arg else 0, n)  # number of repeats
                    args.insert(_arg.index('_conv') - 1 if 'self' in _arg else 0, _conv)
                    n = 1

            elif m in [ChannelAttention, SpatialAttention, CBAM]:
                c2 = ch[f]
                args = [c1, *args]

            elif m in [nn.BatchNorm2d, nn.Identity]:
                args = [ch[f]]

            elif m is Concat:
                c2 = sum(ch[x] for x in f)

            elif m in [vision_backbone, timm_backbone]:
                _arg = inspect.getfullargspec(m)[0]
                if len(args) == 1:
                    args += [False]
                args.insert(_arg.index("in_channels"), ch[f])
                args.insert(_arg.index("num_class"), num_class)
                c2 = num_class
            else:
                c2 = ch[f]

            m_seq = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace("__main__.", "")
            m.np = sum(x.numel() for x in m_seq.parameters())  # number params
            m_seq.i, m_seq.f, m_seq.type = i, ([f] if isinstance(f, int) else f), t  # attach index, 'from' index, type
            #if verbose:
            #    LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print

            save.extend(x for x in ([f] if isinstance(f, int) else f) if x != -1 and x < i)  # append to savelist
            layers.append(m_seq)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save), fc_resize