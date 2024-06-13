from divan import *

if __name__ == '__main__':
    FORMAT = '%(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        format=FORMAT)

    model = DIVAN('resnet34.yaml')
    model.fit('dataset', 50,
              batch_size=128,
              cutmix_p=1,
              label_smoothing=0,
              last_cutmix_close=10,
              lr=0.001,
              EMA=True,
              RAM=True)