from divan import DIVAN


if __name__ == '__main__':
    model = DIVAN()
    model.train('test123.yaml',
                'dataset',
                epochs=2,
                silence=False,
                channels='R',
                channels_mode='auto',
                cuda_idx=0,
                batch_size=64,
                RAM=False,
                amp=True,
                ema=True,
                cutmix_p=1,
                label_smoothing=0.1,
                )
    #custom_model.speed_loader()
