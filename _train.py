from divan import *

if __name__ == '__main__':
    FORMAT = '%(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        format=FORMAT)

    model = DIVAN('resnet34.yaml')
    model.fit('dataset', 100,
              batch_size=128,
              cutmix_p=1,
              label_smoothing=0,
              last_cutmix_close=25,
              lr=0.0003,
              early_stopping=15,
              EMA=True,
              RAM=False)