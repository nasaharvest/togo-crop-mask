import geopandas
import pandas as pd
from pathlib import Path
from pyproj import Transformer

from .base import BaseProcessor

from typing import List, Tuple, Sequence


class TogoProcessor(BaseProcessor):

    dataset = "togo"
    evaluation_dataset = "togo_evaluation"

    def __init__(self, data_folder: Path) -> None:
        super().__init__(data_folder)

    @staticmethod
    def process_single_shapefile(filepath: Path) -> geopandas.GeoDataFrame:
        df = geopandas.read_file(filepath)
        is_crop = 1
        if "non" in filepath.name.lower():
            is_crop = 0

        df["is_crop"] = is_crop

        df["lon"] = df.geometry.centroid.x
        df["lat"] = df.geometry.centroid.y

        df["org_file"] = filepath.name

        return df[["is_crop", "geometry", "lat", "lon", "org_file"]]

    @staticmethod
    def process_eval_shapefile(
        filepaths: Sequence[Tuple[Path, str, bool]]
    ) -> geopandas.GeoDataFrame:

        labels: List[str] = []
        lat_labels: List[str] = []
        lon_labels: List[str] = []
        dfs: List[geopandas.GeoDataFrame] = []

        for idx, (filepath, label, transform_coords) in enumerate(filepaths):
            df = geopandas.read_file(filepath)

            clean_label = f"label{idx}"
            df = df.rename(columns={label: clean_label})
            labels.append(clean_label)

            lat_label, lon_label = "lat", "lon"
            if idx > 0:
                lat_label, lon_label = f"{lat_label}_{idx}", f"{lon_label}_{idx}"

            if transform_coords:
                x = df.geometry.centroid.x.values
                y = df.geometry.centroid.y.values

                transformer = Transformer.from_crs(crs_from=32631, crs_to=4326)

                lat, lon = transformer.transform(xx=x, yy=y)
                df[lon_label] = lon
                df[lat_label] = lat
            else:
                df[lon_label] = df.geometry.centroid.x.values
                df[lat_label] = df.geometry.centroid.y.values

            lat_labels.append(lat_label)
            lon_labels.append(lon_label)
            dfs.append(df[[clean_label, lat_label, lon_label, "id"]])

        df = pd.concat(dfs, axis=1).dropna(how="any")

        # check all the lat labels, lon labels agree
        for i in range(1, len(lon_labels)):
            assert (df[lon_labels[i - 1]] == df[lon_labels[i]]).all()
        for i in range(1, len(lat_labels)):
            assert (df[lat_labels[i - 1]] == df[lat_labels[i]]).all()

        # now, we only want to keep the labels where at least two labellers agreed
        df.loc[:, "sum"] = df[labels].sum(axis=1)

        assert (
            len(filepaths) == 4
        ), f"The logic in process_eval_shapefile assumes 4 labellers"
        df.loc[:, "is_crop"] = 0
        # anywhere where two labellers agreed is crop, we will label as crop
        # this means that rows with 0 or 1 labeller agreeing is crop will be left as non crop
        # so we are always taking the majority
        df.loc[df["sum"] >= 3, "is_crop"] = 1

        # remove ties
        df = df[df["sum"] != 2]

        return df[["is_crop", "lat", "lon"]]

    def process(self, evaluation: bool = True) -> None:

        output_dfs: List[geopandas.GeoDataFrame] = []

        shapefiles = [
            self.raw_folder / "crop_merged_v2",
            self.raw_folder / "noncrop_merged_v2",
        ]

        for filepath in shapefiles:
            output_dfs.append(self.process_single_shapefile(filepath))

        df = pd.concat(output_dfs)
        df["index"] = df.index

        df.to_file(self.output_folder / "data.geojson", driver="GeoJSON")

        if evaluation:

            # the boolean indicates whether or not the coordinates need to
            # be transformed from 32631 to 4326
            evaluation_shapefiles = (
                (self.raw_folder / "random_sample_hrk", "hrk-label", True),
                (self.raw_folder / "random_sample_cn", "cn_labels", False),
                (self.raw_folder / "BB_random_sample_1k", "bb_label", False),
                (self.raw_folder / "random_sample_bm", "bm_labels", False),
            )

            output_folder = self.data_folder / "processed" / self.evaluation_dataset
            output_folder.mkdir(exist_ok=True)

            eval_df = self.process_eval_shapefile(evaluation_shapefiles)

            eval_df["index"] = eval_df.index
            eval_df.to_csv(output_folder / "data.csv", index=False)
