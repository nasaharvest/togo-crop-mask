from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm
import warnings
import xarray as xr

from typing import Dict, List, Tuple, Optional, Union
from src.exporters.sentinel.cloudfree import BANDS
from src.utils import set_seed
from src.utils.regions import BoundingBox


@dataclass
class BaseDataInstance:
    label_lat: float
    label_lon: float
    instance_lat: float
    instance_lon: float
    labelled_array: np.ndarray

    def isin(self, bounding_box: BoundingBox) -> bool:
        return (
            (self.instance_lon <= bounding_box.max_lon)
            & (self.instance_lon >= bounding_box.min_lon)
            & (self.instance_lat <= bounding_box.max_lat)
            & (self.instance_lat >= bounding_box.min_lat)
        )


class BaseEngineer(ABC):
    r"""Combine earth engine sentinel data
    and labels to make numpy arrays which can be input into the
    machine learning models
    """

    sentinel_dataset: str
    dataset: str

    # should be True if the dataset contains data which will
    # only be used for evaluation (e.g. the TogoEvaluation dataset)
    eval_only: bool = False

    def __init__(self, data_folder: Path) -> None:
        set_seed()
        self.data_folder = data_folder
        self.geospatial_files = self.get_geospatial_files(data_folder)
        self.labels = self.read_labels(data_folder)

        self.savedir = self.data_folder / "features" / self.dataset
        self.savedir.mkdir(exist_ok=True, parents=True)

        self.normalizing_dict_interim: Dict[str, Union[np.ndarray, int]] = {"n": 0}

    def get_geospatial_files(self, data_folder: Path) -> List[Path]:
        sentinel_files = data_folder / "raw" / self.sentinel_dataset
        return list(sentinel_files.glob("*.tif"))

    @staticmethod
    @abstractmethod
    def read_labels(data_folder: Path) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    @staticmethod
    def randomly_select_latlon(
        lat: np.ndarray, lon: np.ndarray, label_lat: float, label_lon: float
    ) -> Tuple[float, float]:

        lats = np.random.choice(lat, size=2, replace=False)
        lons = np.random.choice(lon, size=2, replace=False)

        if (lats[0] != label_lat) or (lons[0] != label_lon):
            return lats[0], lons[0]
        else:
            return lats[1], lons[1]

    @staticmethod
    def process_filename(
        filename: str, include_extended_filenames: bool
    ) -> Optional[Tuple[str, datetime, datetime]]:
        r"""
        Given an exported sentinel file, process it to get the start
        and end dates of the data. This assumes the filename ends with '.tif'
        """
        date_format = "%Y-%m-%d"

        identifier, start_date_str, end_date_str = filename[:-4].split("_")

        start_date = datetime.strptime(start_date_str, date_format)

        try:
            end_date = datetime.strptime(end_date_str, date_format)
            return identifier, start_date, end_date

        except ValueError:
            if include_extended_filenames:
                end_list = end_date_str.split("-")
                end_year, end_month, end_day = (
                    end_list[0],
                    end_list[1],
                    end_list[2],
                )

                # if we allow extended filenames, we want to
                # differentiate them too
                id_number = end_list[3]
                identifier = f"{identifier}-{id_number}"

                return (
                    identifier,
                    start_date,
                    datetime(int(end_year), int(end_month), int(end_day)),
                )
            else:
                print(f"Unexpected filename {filename} - skipping")
                return None

    @staticmethod
    def load_tif(
        filepath: Path, start_date: datetime, days_per_timestep: int
    ) -> xr.DataArray:
        r"""
        The sentinel files exported from google earth have all the timesteps
        concatenated together. This function loads a tif files and splits the
        timesteps
        """

        # this mirrors the eo-learn approach
        # also, we divide by 10,000, to remove the scaling factor
        # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2
        da = xr.open_rasterio(filepath).rename("FEATURES") / 10000

        da_split_by_time: List[xr.DataArray] = []

        bands_per_timestep = len(BANDS)
        num_bands = len(da.band)

        assert (
            num_bands % bands_per_timestep == 0
        ), f"Total number of bands not divisible by the expected bands per timestep"

        cur_band = 0
        while cur_band + bands_per_timestep <= num_bands:
            time_specific_da = da.isel(
                band=slice(cur_band, cur_band + bands_per_timestep)
            )
            time_specific_da["band"] = range(bands_per_timestep)
            da_split_by_time.append(time_specific_da)
            cur_band += bands_per_timestep

        timesteps = [
            start_date + timedelta(days=days_per_timestep) * i
            for i in range(len(da_split_by_time))
        ]

        combined = xr.concat(da_split_by_time, pd.Index(timesteps, name="time"))
        combined.attrs["band_descriptions"] = BANDS

        return combined

    def update_normalizing_values(self, array: np.ndarray) -> None:
        # given an input array of shape [timesteps, bands]
        # update the normalizing dict
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # https://www.johndcook.com/blog/standard_deviation/
        num_bands = array.shape[1]

        # initialize
        if "mean" not in self.normalizing_dict_interim:
            self.normalizing_dict_interim["mean"] = np.zeros(num_bands)
            self.normalizing_dict_interim["M2"] = np.zeros(num_bands)

        for time_idx in range(array.shape[0]):

            self.normalizing_dict_interim["n"] += 1

            x = array[time_idx, :]

            delta = x - self.normalizing_dict_interim["mean"]
            self.normalizing_dict_interim["mean"] += (
                delta / self.normalizing_dict_interim["n"]
            )
            self.normalizing_dict_interim["M2"] += delta * (
                x - self.normalizing_dict_interim["mean"]
            )

    def calculate_normalizing_dict(self) -> Optional[Dict[str, np.ndarray]]:

        if "mean" not in self.normalizing_dict_interim:
            print(
                "No normalizing dict calculated! Make sure to call update_normalizing_values"
            )
            return None

        variance = self.normalizing_dict_interim["M2"] / (
            self.normalizing_dict_interim["n"] - 1
        )
        std = np.sqrt(variance)

        return {"mean": self.normalizing_dict_interim["mean"], "std": std}

    @staticmethod
    def maxed_nan_to_num(
        array: np.ndarray, nan: float, max_ratio: Optional[float] = None
    ) -> Optional[np.ndarray]:

        if max_ratio is not None:
            num_nan = np.count_nonzero(np.isnan(array))
            if (num_nan / array.size) > max_ratio:
                return None
        return np.nan_to_num(array, nan=nan)

    @abstractmethod
    def process_single_file(
        self,
        path_to_file: Path,
        nan_fill: float,
        max_nan_ratio: float,
        add_ndvi: bool,
        calculate_normalizing_dict: bool,
        start_date: datetime,
        days_per_timestep: int,
        is_test: bool,
    ) -> Optional[BaseDataInstance]:
        raise NotImplementedError

    @staticmethod
    def calculate_ndvi(input_array: np.ndarray, num_dims: int = 2) -> np.ndarray:
        r"""
        Given an input array of shape [timestep, bands] where
        bands == len(BANDS), returns an array of shape
        [timestep, bands + 1] where the extra band is NDVI,
        (b08 - b04) / (b08 + b04)
        """

        if num_dims == 2:
            near_infrared = input_array[:, BANDS.index("B8")]
            red = input_array[:, BANDS.index("B4")]
        elif num_dims == 3:
            near_infrared = input_array[:, :, BANDS.index("B8")]
            red = input_array[:, :, BANDS.index("B4")]
        else:
            raise ValueError(f"Expected num_dims to be 2 or 3 - got {num_dims}")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in true_divide"
            )
            # suppress the following warning
            # RuntimeWarning: invalid value encountered in true_divide
            # for cases where near_infrared + red == 0
            # since this is handled in the where condition
            ndvi = np.where(
                (near_infrared + red) > 0,
                (near_infrared - red) / (near_infrared + red),
                0,
            )
        return np.append(input_array, np.expand_dims(ndvi, -1), axis=-1)

    def engineer(
        self,
        val_set_size: float = 0.2,
        test_set_size: float = 0.0,
        nan_fill: float = 0.0,
        max_nan_ratio: float = 0.3,
        checkpoint: bool = True,
        add_ndvi: bool = True,
        include_extended_filenames: bool = True,
        calculate_normalizing_dict: bool = True,
        days_per_timestep: int = 30,
    ):
        r"""
        Run the engineer

        :param val_set_size: The ratio of labels which should be put into the validation set
        :param test_set_size: The ratio of labels which should be put into the test set.
            This is 0 by default, since the Togo Evaluation labels are used to calculate
            test results instead
        :param nan_fill: The value to use to fill NaNs
        :param max_nan_ratio: The maximum number of NaNs in an array. Data with more NaNs than
            this is skipped
        :param checkpoint: Whether to check in self.data_folder to see if a file has already been
            engineered. If it is, it is skipped
        :param add_ndvi: Whether to add NDVI to the raw bands
        :param include_extended_filenames: Filenames are expected to have the format
            {identifier}_{start_date}_{end_date}.tif - some have additional strings, i.e.
            {identifier}_{start_date}_{end_date}{more_strings}.tif. If include_extended_filenames
            is True, those files get engineered too. Otherwise, they get skipped
        :param calculate_normalizing_dict: Whether to calculate a normalizing dictionary (i.e. the
            mean and standard deviation of all training and validation files)
        :param days_per_timestep: The number of days per timestep. This should match the value
            passed to the exporter
        """
        for file_path in tqdm(self.geospatial_files):

            file_info = self.process_filename(
                file_path.name, include_extended_filenames=include_extended_filenames
            )

            if file_info is None:
                continue

            identifier, start_date, end_date = file_info

            file_name = f"{identifier}_{str(start_date.date())}_{str(end_date.date())}"

            if checkpoint:
                # we check if the file has already been written
                if (
                    (self.savedir / "validation" / f"{file_name}.pkl").exists()
                    or (self.savedir / "training" / f"{file_name}.pkl").exists()
                    or (self.savedir / "testing" / f"{file_name}.pkl").exists()
                ):
                    continue

            if self.eval_only:
                data_subset = "testing"
            else:
                random_float = np.random.uniform()
                # we split into (val, test, train)
                if random_float <= (val_set_size + test_set_size):
                    if random_float <= val_set_size:
                        data_subset = "validation"
                    else:
                        data_subset = "testing"
                else:
                    data_subset = "training"

            instance = self.process_single_file(
                file_path,
                nan_fill=nan_fill,
                max_nan_ratio=max_nan_ratio,
                add_ndvi=add_ndvi,
                calculate_normalizing_dict=calculate_normalizing_dict,
                start_date=start_date,
                days_per_timestep=days_per_timestep,
                is_test=True if data_subset == "testing" else False,
            )
            if instance is not None:
                subset_path = self.savedir / data_subset
                subset_path.mkdir(exist_ok=True)
                save_path = subset_path / f"{file_name}.pkl"
                with save_path.open("wb") as f:
                    pickle.dump(instance, f)

        if calculate_normalizing_dict:
            normalizing_dict = self.calculate_normalizing_dict()

            if normalizing_dict is not None:
                save_path = self.savedir / "normalizing_dict.pkl"
                with save_path.open("wb") as f:
                    pickle.dump(normalizing_dict, f)
            else:
                print("No normalizing dict calculated!")
