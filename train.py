from divan import *

if __name__ == '__main__':
    FORMAT = '%(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        format=FORMAT)

    model = DIVAN('Multi_channel.yaml')
    model.fit('dataset', 10,
              batch_size=128,
              cutmix_p=0,
              label_smoothing=0.,
              last_cutmix_close=0,
              lr=0.0025, #lr=0.0005, #0.0025
              early_stopping=15,
              EMA=False,
              RAM=False)
