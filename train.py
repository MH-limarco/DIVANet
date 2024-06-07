from divan import DIVAN


if __name__ == "__main__":
    model = DIVAN()
    model.fit("C2f-CBAM-FastKANConv.yaml",
              "dataset",
              epochs=30,
              silence=False,
              channels="RGB",
              channels_mode="smooth", # 'smooth', 'hard', 'auto'
              batch_size=64,
              RAM=False,
              amp=True,
              ema=True,
              label_smoothing=0.1,
              cuda_idx=0,
              num_workers=-1,
              cutmix_p=1)