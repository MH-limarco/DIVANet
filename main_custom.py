from src import *


if __name__ == '__main__':
    custom_model = Custom_CV('test123.yaml',
                           'dataset',
                           silence=False,
                           channels='R',
                           channels_mode='auto',
                           cuda_idx=0,
                           batch_size=64,
                           RAM=True,
                           amp=True,
                           ema=True,
                           cutmix_p=1,
                           label_smoothing=0.1,
                           )

    #custom_model.speed_loader()
    custom_model.train(epochs=2)