import lightning as L
import hydra

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger

from model import BaseModel
from datamodule import OpenEarthMapDataModule


@hydra.main(version_base=None, config_path="config", config_name="debug")
def run(cfg: DictConfig):

    datamodule: OpenEarthMapDataModule = instantiate(config=cfg.datamodule)

    model: BaseModel = instantiate(
        config=cfg.model, cfg=cfg, datamodule=datamodule, _recursive_=False
    )

    logger = WandbLogger(
        name=cfg["experiment"]["name"],
        project=cfg["logger"]["project"],
        save_dir=cfg["logger"]["save_dir"],
        tags=cfg["experiment"]["tags"],
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    trainer: L.Trainer = instantiate(cfg.trainer, logger=logger)

    if cfg["experiment"]["train"]:
        trainer.fit(model, datamodule)

    if cfg["experiment"]["test"]:
        trainer.test(model, datamodule)


if __name__ == "__main__":
    run()
