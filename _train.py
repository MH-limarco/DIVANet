from divan import DIVAN

if __name__ == '__main__':
    #FORMAT = '%(message)s'
    #logging.basicConfig(level=logging.DEBUG,
    #                    format=FORMAT)

    model = DIVAN('Multi_channel.yaml')
    model.fit('dataset', 10, #100
              batch_size=128,
              cutmix_p=0,
              label_smoothing=0.0,
              last_cutmix_close=0,
              lr=0.0025, #=0.0005
              early_stopping=30,
              EMA=False,
              RAM=False)



    #model.fit('dataset', 100,
    #          batch_size=128,
    #          cutmix_p=0,
    #          label_smoothing=0.1,
    #          last_cutmix_close=15,
    #          lr=0.00025, #0.00025 try?
    #          T_mult=2,
    #          early_stopping=15,
    #          EMA=False,
    #          RAM=False)