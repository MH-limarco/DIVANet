from divan import DIVAN_torch

if __name__ == '__main__':
    model = DIVAN_torch()

    model.train('resnet34',
                'dataset',
                pretrained=False,
                epochs=10,
                silence=True,
                channels='RGB',
                channels_mode='hard',
                cuda_idx=0,
                batch_size=128,
                RAM=False,
                amp=True,
                ema=True,
                cutmix_p=1,
                label_smoothing=0.1,
                )

