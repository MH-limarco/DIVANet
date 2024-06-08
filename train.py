from divan import DIVAN


if __name__ == "__main__":
    model = DIVAN()
    model.fit("resnet34.yaml",
              "dataset",
              epochs=100,
              silence=False,
              channels="RGB",
              channels_mode="auto", # 'smooth', 'hard', 'auto'
              batch_size=64,
              RAM=True,
              amp=True,
              ema=True,
              label_smoothing=0.1,
              cuda_idx=None,
              num_workers=-1,
              cutmix_p=1)