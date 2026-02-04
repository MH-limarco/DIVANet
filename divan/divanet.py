import warnings
from typing import Optional

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from divan.core import Task
from divan.core.data import DivanDataModule
from divan.callbacks import YOLOProgressBar, EMAWeightAveraging
from divan.utils import load_yaml, get_logger, configure_lightning_logging, seed_everything

configure_lightning_logging()
log = get_logger(__name__)
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

class DIVAN:
    """
    DIVAN High-Level Facade (v1.0)
    負責協調 Data 和 Model 的依賴關係。
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg = load_yaml(config_path)
        self.trainer = None
        self.engine = None
        self.datamodule = None

    def fit(self,
            data_path: str,  # "MNIST" or "data.yaml"
            epochs: int = 100,
            batch_size: int = 64,
            optimizer: torch.optim.Optimizer = None,
            scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            optimizer_kwargs: dict = None,
            scheduler_kwargs: dict = None,
            warmup_epochs: int = 0,
            ckpt_path: str = None,
            save_top_k: int = 3,

            seed: int = 42,  # Deterministic
            patience: int = 10,
            precision: str = "32",
            use_ema: bool = False,  # EMA
            ema_decay: float = 0.9995,
            grad_clip: float = 0.0,

            fast_dev_run: bool = False,
            **trainer_kwargs):

        train_hparams = {
            "model_path": self.config_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer_name": optimizer if optimizer is None or not hasattr(optimizer, "__name__") else optimizer.__name__,
            "optimizer_kwargs": optimizer_kwargs,
            "scheduler_name": scheduler if scheduler is None or not hasattr(scheduler, "__name__") else scheduler.__name__,
            "scheduler_kwargs": scheduler_kwargs,
            "warmup_epochs": warmup_epochs,
            "patience": patience,
            "precision": precision,
            "seed": seed,
            "use_ema": use_ema,
            "grad_clip": grad_clip
        }

        # 1. 先建立 DataModule
        log.info("Initializing Data Pipeline...")
        if seed is not None:
            seed_everything(seed)

        # 從主設定檔中提取 'data' 區塊
        data_config = self.cfg.get('transforms', {})

        self.datamodule = DivanDataModule(
            data_path=data_path,  # <--- 關鍵：直接傳給 DataModule 判斷
            config=data_config,  # 這裡只剩 transforms 設定
            batch_size=batch_size
        )

        # 關鍵步驟：主動執行 setup 來掃描 Dataset
        self.datamodule.setup_datasets()
        nc = self.datamodule.num_classes

        log.info(f"Dataset scanned. Found {nc} classes.")
        # 2. 再建立 Model Engine (注入 real_nc)
        log.info("Initializing Model Engine...")
        self.engine = Task(
            config=self.cfg,
            in_channels=self.datamodule.in_channels,
            num_classes=nc,
            train_hparams=train_hparams
        )
        # 3. 處理依賴注入 (Optimizer)

        self.engine.set_custom_optimizer(
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
            warmup_epochs=warmup_epochs,
            )

        metric = "val/acc"
        checkpoint_callback = ModelCheckpoint(
            filename=f"{self.cfg.get('name', 'model')}" + "-epoch_{epoch:02d}-metric_{" + metric + ":.3f}",
            monitor=metric,
            mode="max",
            save_last=True,
            save_top_k=save_top_k,
            auto_insert_metric_name=False,
        )

        earlystop_callback = EarlyStopping(monitor="val/loss", patience=patience, mode="min")

        callbacks = trainer_kwargs.pop('callbacks', [])
        if not isinstance(callbacks, list):
            callbacks = [callbacks]

        if use_ema:
            callbacks.append(EMAWeightAveraging(decay=ema_decay))

        callbacks.append(checkpoint_callback)
        callbacks.append(earlystop_callback)
        callbacks.append(YOLOProgressBar())

        # 4. 初始化 Trainer
        self.trainer = L.Trainer(
            max_epochs=epochs,
            fast_dev_run=fast_dev_run,  # 傳入此參數
            callbacks=callbacks,
            precision=precision,
            gradient_clip_val=grad_clip,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=25,
            **trainer_kwargs
        )

        # 5. 開始訓練
        log.info("Starting training...")
        self.trainer.fit(self.engine, datamodule=self.datamodule, ckpt_path=ckpt_path)

    def val(self,
            batch_size: Optional[int] = None,
            ckpt_path: str = "best"):

        if batch_size is not None:
            self.datamodule.batch_size = batch_size

        self.trainer.validate(self.engine, datamodule=self.datamodule, ckpt_path=ckpt_path)

    def test(self,
             batch_size: Optional[int] = None,
             ckpt_path: str = "best"):

        if batch_size is not None:
            self.datamodule.batch_size = batch_size

        self.trainer.test(self.engine, datamodule=self.datamodule, ckpt_path=ckpt_path)

    def predict_step(self, ckpt_path: str = "best",
                     datamodule: Optional[DivanDataModule] = None,):

        if datamodule is None:
            datamodule = self.datamodule

        self.trainer.predict(self.engine, datamodule=datamodule, ckpt_path=ckpt_path)


