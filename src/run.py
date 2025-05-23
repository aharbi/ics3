import lightning as L
import hydra
import logging

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from model import BaseModel
from datamodule import OpenEarthMapDataModule


@hydra.main(version_base=None, config_path="config", config_name="debug")
def run(cfg: DictConfig):

    datamodule: OpenEarthMapDataModule = instantiate(config=cfg.datamodule)

    model: BaseModel = instantiate(config=cfg.model, cfg=cfg, _recursive_=False)

    if cfg.experiment.checkpoint is not None:
        ckpt_path = cfg.experiment.checkpoint
        model = type(model).load_from_checkpoint(
            ckpt_path,
            cfg=cfg,
        )

        logging.info(f"Loaded model from {ckpt_path}")

    logger = WandbLogger(
        name=cfg["experiment"]["name"],
        project=cfg["logger"]["project"],
        save_dir=cfg["logger"]["save_dir"],
        tags=cfg["experiment"]["tags"],
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/mean_iou",
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer: L.Trainer = instantiate(
        cfg.trainer, logger=logger, callbacks=[checkpoint_callback, lr_monitor]
    )

    if cfg["experiment"]["train"]:
        trainer.fit(model, datamodule)

    if cfg["experiment"]["test"]:
        trainer.test(model, datamodule)


if __name__ == "__main__":
    run()
