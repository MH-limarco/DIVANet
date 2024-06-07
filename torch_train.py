from divan import DIVAN_torch

if __name__ == '__main__':
    model = DIVAN_torch()

    model.fit('resnet34',
              'dataset',
              epochs=10,
              silence=False,
              channels='RGB',
              channels_mode='auto',
              batch_size=128,
              RAM=True,
              amp=True,
              ema=True,
              label_smoothing=0.1,
              cuda_idx=0,
              cutmix_p=1,
              pretrained=False)

