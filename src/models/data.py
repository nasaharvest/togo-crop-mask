from pathlib import Path
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset

from src.exporters import GeoWikiExporter
from src.exporters.sentinel.cloudfree import BANDS
from src.processors import TogoProcessor
from src.utils import STR2BB
from src.engineer.togo import TogoDataInstance
from src.engineer.geowiki import GeoWikiDataInstance

from typing import cast, Tuple, Optional, List, Dict, Sequence


class LandTypeClassificationDataset(Dataset):
    r"""
    A dataset for land-type classification data.
    Iterating through this dataset returns a tuple
    (x, y, weight), where weight is an optionally added
    geographically defined weight.

    The dataset should be called through the model - the parameters
    are defined there.
    """

    def __init__(
        self,
        data_folder: Path,
        subset: str,
        crop_probability_threshold: Optional[float],
        include_geowiki: bool,
        include_togo: bool,
        remove_b1_b10: bool,
        normalizing_dict: Optional[Dict] = None,
    ) -> None:

        self.normalizing_dict: Optional[Dict] = None
        self.data_folder = data_folder
        self.features_dir = data_folder / "features"
        self.bands_to_remove = ["B1", "B10"]
        self.remove_b1_b10 = remove_b1_b10

        assert subset in ["training", "validation", "testing"]
        self.subset_name = subset

        self.crop_probability_threshold = crop_probability_threshold

        if (subset == "testing") and (
            self.features_dir / TogoProcessor.evaluation_dataset
        ).exists():
            print("Evaluating using the togo evaluation dataset!")

            assert normalizing_dict is not None
            self.normalizing_dict = normalizing_dict

            self.pickle_files, _ = self.load_files_and_normalizing_dict(
                self.features_dir / TogoProcessor.evaluation_dataset, subset
            )

        else:
            assert (
                max(include_geowiki, include_togo) is True
            ), "At least one dataset must be included"

            files_and_dicts: List[Tuple[List[Path], Optional[Dict]]] = []

            if include_geowiki:

                geowiki_files, geowiki_nd = self.load_files_and_normalizing_dict(
                    self.features_dir / GeoWikiExporter.dataset, self.subset_name
                )

                files_and_dicts.append((geowiki_files, geowiki_nd))

            if include_togo:
                togo_files, togo_nd = self.load_files_and_normalizing_dict(
                    self.features_dir / TogoProcessor.dataset, self.subset_name,
                )
                files_and_dicts.append((togo_files, togo_nd))

            if normalizing_dict is None:
                # if a normalizing dict wasn't passed to the constructor,
                # then we want to make our own
                self.normalizing_dict = self.adjust_normalizing_dict(
                    [(len(x[0]), x[1]) for x in files_and_dicts]
                )
            else:
                self.normalizing_dict = normalizing_dict

            pickle_files: List[Path] = []
            for files, _ in files_and_dicts:
                pickle_files.extend(files)
            self.pickle_files = pickle_files

    @property
    def num_output_classes(self) -> int:
        return 1

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the anchor, neighbour, distant tensors
        """
        target_file = self.pickle_files[index]

        # first, we load up the target file
        with target_file.open("rb") as f:
            target_datainstance = pickle.load(f)

        if isinstance(target_datainstance, GeoWikiDataInstance):
            if self.crop_probability_threshold is None:
                label = target_datainstance.crop_probability
            else:
                label = int(
                    target_datainstance.crop_probability
                    >= self.crop_probability_threshold
                )
        elif isinstance(target_datainstance, TogoDataInstance):
            label = target_datainstance.is_crop
        else:
            raise RuntimeError(
                f"Unrecognized data instance type {type(target_datainstance)}"
            )

        weight = 0
        if target_datainstance.isin(STR2BB["Togo"]):
            weight = 1

        return (
            torch.from_numpy(
                self.remove_bands(x=self._normalize(target_datainstance.labelled_array))
            ).float(),
            torch.tensor(label).float(),
            torch.tensor(weight).long(),
        )

    @staticmethod
    def adjust_normalizing_dict(
        dicts: Sequence[Tuple[int, Optional[Dict[str, np.ndarray]]]]
    ) -> Optional[Dict[str, np.ndarray]]:

        for length, single_dict in dicts:
            if single_dict is None:
                return None

        dicts = cast(Sequence[Tuple[int, Dict[str, np.ndarray]]], dicts)

        new_total = sum([x[0] for x in dicts])

        new_mean = (
            sum([single_dict["mean"] * length for length, single_dict in dicts])
            / new_total
        )

        new_variance = (
            sum(
                [
                    (single_dict["std"] ** 2 + (single_dict["mean"] - new_mean) ** 2)
                    * length
                    for length, single_dict in dicts
                ]
            )
            / new_total
        )

        return {"mean": new_mean, "std": np.sqrt(new_variance)}

    @property
    def num_input_features(self) -> int:
        # assumes the first value in the tuple is x
        assert len(self.pickle_files) > 0, "No files to load!"
        output_tuple = self[0]
        return output_tuple[0].shape[1]

    @property
    def num_timesteps(self) -> int:
        # assumes the first value in the tuple is x
        assert len(self.pickle_files) > 0, "No files to load!"
        output_tuple = self[0]
        return output_tuple[0].shape[0]

    def remove_bands(self, x: np.ndarray) -> np.ndarray:
        """
        Expects the input to be of shape [timesteps, bands]
        """
        if self.remove_b1_b10:
            indices_to_remove: List[int] = []
            for band in self.bands_to_remove:
                indices_to_remove.append(BANDS.index(band))

            bands_index = 1 if len(x.shape) == 2 else 2
            indices_to_keep = [
                i for i in range(x.shape[bands_index]) if i not in indices_to_remove
            ]
            if len(x.shape) == 2:
                # timesteps, bands
                return x[:, indices_to_keep]
            else:
                # batches, timesteps, bands
                return x[:, :, indices_to_keep]
        else:
            return x

    @staticmethod
    def load_files_and_normalizing_dict(
        features_dir: Path, subset_name: str
    ) -> Tuple[List[Path], Optional[Dict[str, np.ndarray]]]:
        pickle_files = list((features_dir / subset_name).glob("*.pkl"))

        # try loading the normalizing dict. By default, if it exists we will use it
        if (features_dir / "normalizing_dict.pkl").exists():
            with (features_dir / "normalizing_dict.pkl").open("rb") as f:
                normalizing_dict = pickle.load(f)
        else:
            normalizing_dict = None

        return pickle_files, normalizing_dict

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        if self.normalizing_dict is None:
            return array
        else:
            return (array - self.normalizing_dict["mean"]) / self.normalizing_dict[
                "std"
            ]

    def __len__(self) -> int:
        return len(self.pickle_files)
