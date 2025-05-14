import os
import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import Point


def prediction_figure(
    satellite_image: np.array, prediction: np.array, ground_truth: np.array
):

    satellite_image = satellite_image.transpose(1, 2, 0)

    satellite_image = (satellite_image * 255).astype(np.uint8)
    prediction = (prediction * 255).astype(np.uint8)
    ground_truth = (ground_truth * 255).astype(np.uint8)

    prediction = prediction.squeeze()
    ground_truth = ground_truth.squeeze()

    image = Image.fromarray(satellite_image)
    prediction = Image.fromarray(prediction, mode="L")
    ground_truth = Image.fromarray(ground_truth, mode="L")

    return image, prediction, ground_truth


def joint_prediction_figure(
    satellite_image: np.array, prediction: np.array, ground_truth: np.array = None
):

    satellite_image = satellite_image.transpose(1, 2, 0)
    prediction = prediction.transpose(1, 2, 0)

    if ground_truth is not None:
        ground_truth = ground_truth.transpose(1, 2, 0)

    satellite_image = satellite_image.astype(np.float32)
    prediction = prediction.astype(np.float32)
    if ground_truth is not None:
        ground_truth = ground_truth.astype(np.float32)

    fig, axes = plt.subplots(1, 3 if ground_truth is not None else 2, figsize=(15, 5))

    axes[0].imshow(satellite_image)
    axes[0].set_title("Satellite Image")
    axes[0].axis("off")

    axes[1].imshow(prediction)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    if ground_truth is not None:
        axes[2].imshow(ground_truth)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    return fig


def get_dataset_lat_lon(dataset_path: Path):

    regions = os.listdir(dataset_path)

    regions_data = {}

    for region in regions:
        region_path = dataset_path / region / "images"

        if not region_path.is_dir():
            continue

        images = os.listdir(region_path)

        num_images = len(images)

        if num_images == 0:
            print(f"No images found in {region_path}")
            continue

        image = images[0]

        image_path = region_path / image
        with rasterio.open(image_path) as src:
            metadata = src.meta
            width = metadata["width"]
            height = metadata["height"]
            crs = metadata["crs"]
            transform = metadata["transform"]

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        row = height // 2
        col = width // 2

        x, y = transform * (col, row)

        lon, lat = transformer.transform(x, y)

        regions_data[region] = {
            "num_images": num_images,
            "latitude": lat,
            "longitude": lon,
        }

    return regions_data


def regions_to_geojson(regions_data: dict, output_path: Path):
    data = []
    for region, info in regions_data.items():
        point = Point(info["longitude"], info["latitude"])
        data.append(
            {"region": region, "num_images": info["num_images"], "geometry": point}
        )

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    output_path = output_path / "regions.geojson"

    gdf.to_file(output_path, driver="GeoJSON")


if __name__ == "__main__":
    dataset_path = Path("/Users/aziz/git/ics3/data/OpenEarthMap_wo_xBD/")
    output_path = Path("/Users/aziz/git/ics3/data/")

    regions = get_dataset_lat_lon(dataset_path)
    regions_to_geojson(regions, output_path)
