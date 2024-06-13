from divan import *

if __name__ == '__main__':
    FORMAT = '%(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        format=FORMAT)

    model = DIVAN('resnet34.yaml')
    model.fit('dataset', 1,
              batch_size=128,
              RAM=False)