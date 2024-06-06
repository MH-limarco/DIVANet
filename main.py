from src import *

if __name__ == '__main__':
    torch_model = Torch_CV()

    torch_model.train('resnet34',
                           'dataset',
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
                           pretrained=False)

