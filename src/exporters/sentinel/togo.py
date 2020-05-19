import pandas as pd
import geopandas
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta, date

from .base import BaseSentinelExporter
from src.processors.togo import TogoProcessor
from .utils import EEBoundingBox, bounding_box_from_centre

from typing import Optional, List


class TogoSentinelExporter(BaseSentinelExporter):

    dataset = "earth_engine_togo"
    evaluation_dataset = "earth_engine_togo_evaluation"

    # We will use the same data collection date
    # as the kenya non crop data for consistency.
    # This data was actually collected on the
    # 5 May 2020
    data_date = date(2020, 4, 16)

    def load_labels(self) -> pd.DataFrame:
        # right now, this just loads geowiki data. In the future,
        # it would be neat to merge all labels together
        togo = self.data_folder / "processed" / TogoProcessor.dataset / "data.geojson"
        assert togo.exists(), "Togo processor must be run to load labels"
        return geopandas.read_file(togo)[["lat", "lon"]]

    def load_evaluation_labels(self) -> pd.DataFrame:

        data = (
            self.data_folder
            / "processed"
            / TogoProcessor.evaluation_dataset
            / "data.csv"
        )
        assert (
            data.exists()
        ), "Togo processor must be run on the evaluation data to load labels"
        return pd.read_csv(data)[["lat", "lon"]]

    def labels_to_bounding_boxes(
        self, num_labelled_points: Optional[int], surrounding_metres: int
    ) -> List[EEBoundingBox]:

        output: List[EEBoundingBox] = []

        for idx, row in tqdm(self.labels.iterrows()):

            output.append(
                bounding_box_from_centre(
                    mid_lat=row["lat"],
                    mid_lon=row["lon"],
                    surrounding_metres=surrounding_metres,
                ),
            )

            if num_labelled_points is not None:
                if len(output) >= num_labelled_points:
                    return output
        return output

    def export_for_labels(
        self,
        days_per_timestep: int = 30,
        num_timesteps: int = 12,
        num_labelled_points: Optional[int] = None,
        surrounding_metres: int = 80,
        checkpoint: bool = True,
        monitor: bool = False,
        evaluation_set: bool = False,
    ) -> None:
        r"""
        Run the Togo exporter.

        :param days_per_timestep: The number of days of data to use for each mosaiced image.
        :param num_timesteps: The number of timesteps to export
        :param num_labelled_points: (Optional) The number of labelled points to export.
        :param surrounding_metres: The number of metres surrounding each labelled point to export
        :param checkpoint: Whether or not to check in self.data_folder to see if the file has
            already been exported. If it has, skip it
        :param monitor: Whether to monitor each task until it has been run
        :param evaluation_set: Whether or not to export the evaluation set as well
        """

        org_dataset = ""
        org_output_folder = Path(".")
        if evaluation_set:
            # we will override the labels and the dataset
            self.labels = self.load_evaluation_labels()
            org_dataset = self.dataset
            org_output_folder = self.output_folder

            self.dataset = self.evaluation_dataset
            self.output_folder = self.data_folder / "raw" / self.dataset
            self.output_folder.mkdir(parents=True, exist_ok=True)

            print(f"Exporting files to {self.dataset} in earth engine")

        bounding_boxes_to_download = self.labels_to_bounding_boxes(
            num_labelled_points=num_labelled_points,
            surrounding_metres=surrounding_metres,
        )

        start_date = self.data_date - num_timesteps * timedelta(days=days_per_timestep)

        for idx, bounding_info in enumerate(bounding_boxes_to_download):
            self._export_for_polygon(
                polygon=bounding_info.to_ee_polygon(),
                polygon_identifier=idx,
                start_date=start_date,
                end_date=self.data_date,
                days_per_timestep=days_per_timestep,
                checkpoint=checkpoint,
                monitor=monitor,
            )

        if evaluation_set:
            # reset everything to be as expected, just
            # in case
            self.labels = self.load_labels()
            self.dataset = org_dataset
            self.output_folder = org_output_folder
