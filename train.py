from divan import DIVAN

if __name__ == '__main__':
    model = DIVAN('resnet34.yaml',
                  )
    #model = DIVAN('divan_result/train/resnet34-1/weight/last.pt')
    model.fit('dataset',
              epochs=100,
              warnup_step=5,
              endstep_epochs=20,
              endstep_patience=3,
              endstep_factor=0.5,
              batch_size=128,
              cutmix_p=0,
              label_smoothing=0.0,
              lr=0.0005,
              early_stopping=48,
              RAM=True)
