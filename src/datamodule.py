import os
import glob
import lightning as L
import torch
import numpy as np
import rasterio
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from constants import class_label


class OpenEarthMapDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        patch_size: int = 256,
        number_of_samples: int = 1000,
        number_of_context_samples: int = 0,
        downsample: int = 2,
        classes: list[str] = ["Road"],
        regions: list[str] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

        self.patch_size = patch_size
        self.downsample = downsample
        self.number_of_samples = number_of_samples
        self.number_of_context_samples = number_of_context_samples
        self.classes = classes
        self.regions = regions

        self.files = self.initialize_dataset_files()
        self.classes_idx = [class_label[c] for c in self.classes]

    def initialize_dataset_files(self):

        files = dict(list())

        for region in self.regions:
            region_path = os.path.join(self.data_dir, region)

            images_path = os.path.join(region_path, "images")
            labels_path = os.path.join(region_path, "labels")

            images = glob.glob(os.path.join(images_path, "*.tif"))
            labels = glob.glob(os.path.join(labels_path, "*.tif"))

            images = [os.path.basename(f) for f in images]
            labels = [os.path.basename(f) for f in labels]

            # Exclude files with no labels
            region_files = list(set(images) & set(labels))

            files[region] = region_files

        return files

    def load_satellite_image(self, image_path: str):

        with rasterio.open(image_path) as src:
            image = src.read()

        image = image / 255.0

        return image

    def load_label(self, label_path: str):

        with rasterio.open(label_path) as src:
            label = src.read()

        # TODO: Binary classification for the moment
        label = np.isin(label, self.classes_idx).astype(np.uint8)

        return label

    def slice_sample(self, image, label):
        height, width = image.shape[1:]

        x = np.random.randint(low=0, high=height - self.patch_size)
        y = np.random.randint(low=0, high=width - self.patch_size)

        image = image[:, x : x + self.patch_size, y : y + self.patch_size]
        label = label[:, x : x + self.patch_size, y : y + self.patch_size]

        return image, label
    
    def downsample_tensor(self, tensor, factor=2):
        tensor = torch.from_numpy(tensor)
        downsampled = F.interpolate(
            tensor.unsqueeze(0), scale_factor=1 / factor, mode="bilinear", align_corners=False
        )
        downsampled = downsampled.squeeze(0)
        downsampled = downsampled.numpy()
        return downsampled

    def select_random_samples(self, region=None):
        # TODO: Weight probability of each region per number of samples
        if region is None:
            region = np.random.choice(list(self.files.keys()))

        region_files = self.files[region]

        idx = np.random.choice(len(region_files))

        file = region_files[idx]

        region_path = os.path.join(self.data_dir, region)

        image_path = os.path.join(region_path, "images", file)
        label_path = os.path.join(region_path, "labels", file)

        image = self.load_satellite_image(image_path=image_path)
        label = self.load_label(label_path=label_path)

        image, label = self.slice_sample(image=image, label=label)

        # TODO: Check if this is necessary
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        image = self.downsample_tensor(image, factor=self.downsample)
        label = self.downsample_tensor(label, factor=self.downsample)

        return image, label, region

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, idx):
        image, label, region = self.select_random_samples()

        context_set_images = []
        context_set_labels = []

        for _ in range(self.number_of_context_samples):
            context_image, context_label, _ = self.select_random_samples(region=region)
            context_set_images.append(context_image)
            context_set_labels.append(context_label)

        if len(context_set_images) > 0:
            context_set_images = np.stack(context_set_images, axis=0)
            context_set_labels = np.stack(context_set_labels, axis=0)

        return {
            "satellite_image": image,
            "label": label,
            "region": region,
            "context_set_images": context_set_images,
            "context_set_labels": context_set_labels,
        }


class OpenEarthMapDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        patch_size: int = 256,
        downsample: int = 2,
        number_of_samples: dict = {"train": 1000, "val": 100, "test": 100},
        number_of_context_samples: dict = {"train": 0, "val": 0, "test": 0},
        classes: list[str] = ["Road"],
        regions: dict = None,
        batch_size: int = 8,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

        # Dataset parameters
        self.patch_size = patch_size
        self.downsample = downsample
        self.number_of_samples = number_of_samples
        self.number_of_context_samples = number_of_context_samples
        self.classes = classes
        self.regions = regions

        # DataLoader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str):

        self.train_dataset = OpenEarthMapDataset(
            data_dir=self.data_dir,
            patch_size=self.patch_size,
            downsample=self.downsample,
            number_of_samples=self.number_of_samples["train"],
            number_of_context_samples=self.number_of_context_samples["train"],
            classes=self.classes,
            regions=self.regions["train"],
        )

        self.val_dataset = OpenEarthMapDataset(
            data_dir=self.data_dir,
            patch_size=self.patch_size,
            downsample=self.downsample,
            number_of_samples=self.number_of_samples["val"],
            number_of_context_samples=self.number_of_context_samples["val"],
            classes=self.classes,
            regions=self.regions["val"],
        )

        self.test_dataset = OpenEarthMapDataset(
            data_dir=self.data_dir,
            patch_size=self.patch_size,
            downsample=self.downsample,
            number_of_samples=self.number_of_samples["test"],
            number_of_context_samples=self.number_of_context_samples["test"],
            classes=self.classes,
            regions=self.regions["test"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )
