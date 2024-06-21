from divan import *

if __name__ == '__main__':
    FORMAT = '%(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        format=FORMAT)

    model = DIVAN('resnet34.yaml')
    #model = DIVAN('divan_result/train/resnet34-1/weight/last.pt')
    model.fit('dataset', 0,
              warnup_step=5,
              endstep_epochs=25,
              endstep_patience=5,
              endstep_factor=0.5,
              batch_size=128,
              cutmix_p=1,
              label_smoothing=0.0,
              lr=0.00075, #lr=0.0005, #0.0025
              early_stopping=24,
              RAM=False)
