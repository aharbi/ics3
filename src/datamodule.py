import os
import glob
import lightning as L
import numpy as np
import rasterio

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from constants import class_label


class OpenEarthMapDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        patch_size: int = 256,
        number_of_samples: int = 1000,
        classes: list[str] = ["Road"],
        regions: list[str] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

        self.patch_size = patch_size
        self.number_of_samples = number_of_samples
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

    def select_random_samples(self, files, number_of_samples=1):
        # TODO: Weight probability of each region per number of samples
        region = np.random.choice(list(files.keys()))
        region_files = files[region]

        idx = np.random.choice(len(region_files), size=number_of_samples)

        # TODO: Return multiple samples
        file = region_files[idx[0]]
        return file, region

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, idx):
        file, region = self.select_random_samples(self.files, number_of_samples=1)

        region_path = os.path.join(self.data_dir, region)

        image_path = os.path.join(region_path, "images", file)
        label_path = os.path.join(region_path, "labels", file)

        image = self.load_satellite_image(image_path=image_path)
        label = self.load_label(label_path=label_path)

        image, label = self.slice_sample(image=image, label=label)

        # TODO: Check if this is necessary
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image, label


class OpenEarthMapDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        patch_size: int = 256,
        number_of_samples: int = 1000,
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
        self.number_of_samples = number_of_samples
        self.classes = classes
        self.regions = regions

        # DataLoader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        self.prepare_data()
        self.setup(stage="fit")

    def setup(self, stage: str):

        self.train_dataset = OpenEarthMapDataset(
            data_dir=self.data_dir,
            patch_size=self.patch_size,
            number_of_samples=self.number_of_samples,
            classes=self.classes,
            regions=self.regions["train"],
        )

        self.val_dataset = OpenEarthMapDataset(
            data_dir=self.data_dir,
            patch_size=self.patch_size,
            number_of_samples=self.number_of_samples,
            classes=self.classes,
            regions=self.regions["val"],
        )

        self.test_dataset = OpenEarthMapDataset(
            data_dir=self.data_dir,
            patch_size=self.patch_size,
            number_of_samples=self.number_of_samples,
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
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    data_dir = Path("data/OpenEarthMap_wo_xBD")
    dataset = OpenEarthMapDataset(
        data_dir=data_dir,
        patch_size=200,
        number_of_samples=10,
        classes=["Road", "Building"],
        regions=["aachen"],
    )

    import matplotlib.pyplot as plt

    image, label = dataset[0]

    image = image.transpose(1, 2, 0)
    label = label.transpose(1, 2, 0)[:, : , 0]

    from PIL import Image

    image = Image.fromarray((image * 255).astype(np.uint8))
    label = Image.fromarray((label * 255).astype(np.uint8), mode="L")
    image = image.resize((512, 512))
    label = label.resize((512, 512))

    image.save("image.png")
    label.save("label.png")

