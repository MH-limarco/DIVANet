from src import *


if __name__ == '__main__':
    custom_model = Custom_CV('C2f-CBAM.yaml',
                           'dataset',
                           silence=False,
                           channels='R',
                           channels_mode='auto',
                           cuda_idx=1,
                           batch_size=64,
                           RAM=True,
                           amp=True,
                           ema=True,
                           cutmix_p=1,
                           label_smoothing=0.1,
                           )
    custom_model.train(epochs=2)