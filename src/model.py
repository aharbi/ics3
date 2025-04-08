import lightning as L

from omegaconf import DictConfig
from hydra.utils import instantiate

from src.utils import prediction_figure


class BaseModel(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig = None,
        datamodule: L.LightningDataModule = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.datamodule = datamodule

        if cfg is not None:
            self.loss_fn = instantiate(cfg.loss)

    def loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log(name="train/loss", value=loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log(name="val/loss", value=loss, on_step=True)

        if batch_idx == 0:
            satellite_image = x[0].half().cpu().numpy()
            prediction = y_hat[0].half().cpu().numpy()
            ground_truth = y[0].half().cpu().numpy()

            fig = prediction_figure(
                satellite_image=satellite_image,
                prediction=prediction,
                ground_truth=ground_truth,
            )

            self.logger.log_image(
                key="val/predictions",
                images=[fig],
            )

        return loss

    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())
