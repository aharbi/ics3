import lightning as L

from pathlib import Path
from omegaconf import DictConfig
from torch import Tensor

from hydra.utils import instantiate
from torchmetrics import MetricCollection
from torchmetrics.segmentation import MeanIoU

from src.utils import joint_prediction_figure, prediction_figure


class BaseModel(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig = None,
    ):
        super().__init__()
        self.cfg = cfg

        self.val_regions = self.cfg.datamodule.regions.val
        self.test_regions = self.cfg.datamodule.regions.test

        self.val_metric = MeanIoU(num_classes=2)
        self.test_metric = MeanIoU(num_classes=2)

        self.region_wise_val_metric = MetricCollection(
            {r: MeanIoU(num_classes=2) for r in self.val_regions},
            prefix="val/mean_iou/",
        )
        self.region_wise_test_metric = MetricCollection(
            {r: MeanIoU(num_classes=2) for r in self.test_regions},
            prefix="test/mean_iou/",
        )

        if cfg is not None:
            self.loss_fn = instantiate(cfg.loss)

    def predict(self, x: Tensor, context_set: list = None):
        y_hat = self(x)
        return y_hat

    def loss(self, y_hat: Tensor, y: Tensor):
        return self.loss_fn(y_hat, y)

    def training_step(self, batch, batch_idx):
        x = batch["satellite_image"]
        y = batch["label"]
        context_set = batch["context_set"]

        y_hat = self.predict(x=x, context_set=context_set)

        if type(y_hat) is list:
            loss = 0
            for i in range(len(y_hat) - 1):
                y_i = context_set[i][1]
                loss += self.loss(y_hat[i], y_i)
            loss += self.loss(y_hat[-1], y)
        else:
            loss = self.loss(y_hat, y)

        batch_size = x.shape[0]

        self.log(name="train/loss", value=loss, on_step=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluation_step(
            batch,
            batch_idx,
            metric=self.val_metric,
            region_wise_metric=self.region_wise_val_metric,
            split="val",
        )

    def test_step(self, batch, batch_idx):
        self.evaluation_step(
            batch,
            batch_idx,
            metric=self.test_metric,
            region_wise_metric=self.region_wise_test_metric,
            split="test",
            log_predictions=True,
        )

    def evaluation_step(
        self,
        batch,
        batch_idx,
        metric: callable,
        region_wise_metric: dict,
        split: str,
        log_predictions: bool = False,
    ):
        x = batch["satellite_image"]
        y = batch["label"]
        context_set = batch["context_set"]
        regions = batch["region"]

        y_hat = self.predict(x=x, context_set=context_set)

        if type(y_hat) is list:
            y_hat = y_hat[-1]

        loss = self.loss(y_hat, y)

        y_hat_binary = (y_hat > 0.5).long()
        y_binary = (y > 0.5).long()

        # Region-wise mean IoU
        regions_unique = list(set(regions))

        for region in regions_unique:
            region_mask = [r == region for r in regions]
            y_hat_region = y_hat_binary[region_mask]
            y_region = y_binary[region_mask]

            batch_size = y_hat_region.shape[0]

            region_wise_metric[region](y_hat_region, y_region)

            self.log(
                name=f"{split}/mean_iou/{region}",
                value=region_wise_metric[region],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        # Total mean IoU
        metric(y_hat_binary, y_binary)

        batch_size = y_binary.shape[0]

        self.log(
            name=f"{split}/mean_iou",
            value=metric,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            name=f"{split}/loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        if batch_idx == 0:
            satellite_image = x[0].half().cpu().numpy()
            prediction = y_hat[0].half().cpu().numpy()
            ground_truth = y[0].half().cpu().numpy()

            fig = joint_prediction_figure(
                satellite_image=satellite_image,
                prediction=prediction,
                ground_truth=ground_truth,
            )

            self.logger.log_image(
                key=f"{split}/predictions",
                images=[fig],
            )

        if log_predictions:
            log_path = Path(self.cfg.trainer.default_root_dir)
            experiment_name = self.cfg.experiment.name

            log_path = log_path / f"predictions/{experiment_name}/{split}/"

            if not log_path.exists():
                log_path.mkdir(parents=True, exist_ok=True)

            # Log every image
            for i in range(len(x)):
                satellite_image = x[i].half().cpu().numpy()
                prediction = y_hat[i].half().cpu().numpy()
                ground_truth = y[i].half().cpu().numpy()
                region = regions[i].item()

                image, prediction, ground_truth = prediction_figure(
                    satellite_image=satellite_image,
                    prediction=prediction,
                    ground_truth=ground_truth,
                )

                path_image = log_path / f"{region}_{batch_idx}_{i}_image.png"
                path_prediction = log_path / f"{region}_{batch_idx}_{i}_prediction.png"
                path_ground_truth = (
                    log_path / f"{region}_{batch_idx}_{i}_ground_truth.png"
                )

                image.save(path_image)
                prediction.save(path_prediction)
                ground_truth.save(path_ground_truth)

    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())
