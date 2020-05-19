from argparse import ArgumentParser, Namespace
from pathlib import Path
import numpy as np
import xarray as xr
from tqdm import tqdm

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.utils import set_seed
from .model_bases import STR2BASE
from .data import LandTypeClassificationDataset
from .utils import tif_to_np, preds_to_xr

from typing import cast, Callable, Tuple, Dict, Any, Type, Optional, List, Union


class LandCoverMapper(pl.LightningModule):
    r"""
    An LSTM based model to predict the presence of cropland
    inside a pixel.

    hparams
    --------
    The default values for these parameters are set in add_model_specific_args

    :param hparams.data_folder: The path to the data. Default (assumes the model
        is being run from the scripts directory) = "../data"
    :param hparams.model_base: The model base to use. Currently, only an LSTM
        is implemented. Default = "lstm"
    :param hparams.hidden_vector_size: The size of the hidden vector. Default = 64
    :param hparams.learning_rate: The learning rate. Default = 0.001
    :param hparams.batch_size: The batch size. Default = 64
    :param hparams.probability_threshold: The probability threshold to use to label GeoWiki
        instances as crop / not_crop (since the GeoWiki labels are a mean crop probability, as
        assigned by several labellers). In addition, this is the threshold used when calculating
        metrics which require binary predictions, such as accuracy score. Default = 0.5
    :param hparams.num_classification_layers: The number of classification layers to add after
        the base. Default = 1
    :param hparams.classification_dropout: Dropout ratio to apply on the hidden vector before
        it is passed to the classification layer(s). Default = 0
    :param hparams.alpha: The weight to use when adding the global and local losses. This parameter
        is only used if hparams.multi_headed is True. Default = 10
    :param hparams.add_togo: Whether or not to use the hand labelled dataset to train the model.
        Default = True
    :param hparams.add_geowiki: Whether or not to use the GeoWiki dataset to train the model.
        Default = True
    :param hparams.remove_b1_b10: Whether or not to remove the B1 and B10 bands. Default = True
    :param hparams.multi_headed: Whether or not to add a local head, to classify instances within
        Togo. If False, the same classification layer will be used to classify
        all pixels. Default = True
    """

    def __init__(self, hparams: Namespace) -> None:
        super().__init__()

        set_seed()
        self.hparams = hparams

        self.data_folder = Path(hparams.data_folder)

        dataset = self.get_dataset(subset="training")
        self.input_size = dataset.num_input_features
        self.num_outputs = dataset.num_output_classes

        # we save the normalizing dict because we calculate weighted
        # normalization values based on the datasets we combine.
        # The number of instances per dataset (and therefore the weights) can
        # vary between the train / test / val sets - this ensures the normalizing
        # dict stays constant between them
        self.normalizing_dict = dataset.normalizing_dict

        self.model_base_name = hparams.model_base

        self.base = STR2BASE[hparams.model_base](
            input_size=self.input_size, hparams=self.hparams
        )

        global_classification_layers: List[nn.Module] = []
        for i in range(hparams.num_classification_layers):
            global_classification_layers.append(
                nn.Linear(
                    in_features=hparams.hidden_vector_size,
                    out_features=self.num_outputs
                    if i == (hparams.num_classification_layers - 1)
                    else hparams.hidden_vector_size,
                )
            )
            if i < (hparams.num_classification_layers - 1):
                global_classification_layers.append(nn.ReLU())

        self.global_classifier = nn.Sequential(*global_classification_layers)

        if self.hparams.multi_headed:

            local_classification_layers: List[nn.Module] = []
            for i in range(hparams.num_classification_layers):
                local_classification_layers.append(
                    nn.Linear(
                        in_features=hparams.hidden_vector_size,
                        out_features=self.num_outputs
                        if i == (hparams.num_classification_layers - 1)
                        else hparams.hidden_vector_size,
                    )
                )
                if i < (hparams.num_classification_layers - 1):
                    local_classification_layers.append(nn.ReLU())

            self.local_classifier = nn.Sequential(*local_classification_layers)

        self.loss_function: Callable = F.binary_cross_entropy

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        base = self.base(x)
        x_global = self.global_classifier(base)

        if self.num_outputs == 1:
            x_global = torch.sigmoid(x_global)

        if self.hparams.multi_headed:
            x_local = self.local_classifier(base)
            if self.num_outputs == 1:
                x_local = torch.sigmoid(x_local)
            return x_global, x_local

        else:
            return x_global

    def get_dataset(
        self, subset: str, normalizing_dict: Optional[Dict] = None
    ) -> LandTypeClassificationDataset:
        return LandTypeClassificationDataset(
            data_folder=self.data_folder,
            subset=subset,
            crop_probability_threshold=self.hparams.probability_threshold,
            include_geowiki=self.hparams.add_geowiki,
            include_togo=self.hparams.add_togo,
            normalizing_dict=normalizing_dict,
            remove_b1_b10=self.hparams.remove_b1_b10,
        )

    def train_dataloader(self):
        return DataLoader(
            self.get_dataset(subset="training"),
            shuffle=True,
            batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.get_dataset(
                subset="validation", normalizing_dict=self.normalizing_dict
            ),
            batch_size=self.hparams.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.get_dataset(subset="testing", normalizing_dict=self.normalizing_dict),
            batch_size=self.hparams.batch_size,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        return self._split_preds_and_get_loss(
            batch, add_preds=False, loss_label="loss", log_loss=True
        )

    def validation_step(self, batch, batch_idx):
        return self._split_preds_and_get_loss(
            batch, add_preds=True, loss_label="val_loss", log_loss=False
        )

    def test_step(self, batch, batch_idx):
        return self._split_preds_and_get_loss(
            batch, add_preds=True, loss_label="test_loss", log_loss=False
        )

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        tensorboard_logs = {"val_loss": avg_loss}
        tensorboard_logs.update(self.get_interpretable_metrics(outputs, prefix="val_"))

        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean().item()

        output_dict = {"test_loss": avg_loss}
        output_dict.update(self.get_interpretable_metrics(outputs, prefix="test_"))

        return {"progress_bar": output_dict}

    def save_validation_predictions(self) -> None:
        """
        This can be useful in combination with src.utils.plot_roc_curve
        to find an appropriate threshold.
        """
        save_dir = (
            Path(self.hparams.data_folder) / self.__class__.__name__ / "validation"
        )
        save_dir.mkdir(exist_ok=True, parents=True)

        val_dl = self.val_dataloader()

        outputs: List[Dict] = []
        for idx, batch in enumerate(val_dl):

            with torch.no_grad():
                outputs.append(self.validation_step(batch, idx))

        all_preds = (torch.cat([x["pred"] for x in outputs]).detach().cpu().numpy(),)
        all_labels = (torch.cat([x["label"] for x in outputs]).detach().cpu().numpy(),)

        np.save(save_dir / "all_preds.npy", all_preds)
        np.save(save_dir / "all_labels.npy", all_labels)

        if self.hparams.multi_headed:
            r_preds = (
                torch.cat([x["Togo_pred"] for x in outputs]).detach().cpu().numpy()
            )
            r_labels = (
                torch.cat([x["Togo_label"] for x in outputs]).detach().cpu().numpy()
            )

            np.save(save_dir / "Togo_preds.npy", r_preds)
            np.save(save_dir / "Togo_labels.npy", r_labels)

    def predict(
        self,
        path_to_file: Path,
        batch_size: int = 64,
        add_ndvi: bool = True,
        nan_fill: float = 0,
        days_per_timestep: int = 30,
        local_head: bool = True,
    ) -> xr.Dataset:

        self.eval()

        input_data = tif_to_np(
            path_to_file,
            add_ndvi=add_ndvi,
            nan=nan_fill,
            normalizing_dict=self.normalizing_dict,
            days_per_timestep=days_per_timestep,
        )

        dataset = self.get_dataset(subset="training")

        predictions: List[np.ndarray] = []
        cur_i = 0

        pbar = tqdm(total=input_data.x.shape[0] - 1)
        while cur_i < (input_data.x.shape[0] - 1):
            batch_x = torch.from_numpy(
                dataset.remove_bands(input_data.x[cur_i : cur_i + batch_size])
            ).float()

            with torch.no_grad():
                batch_preds = self.forward(batch_x)

                if self.hparams.multi_headed:
                    global_preds, local_preds = batch_preds

                    if local_head:
                        batch_preds = local_preds
                    else:
                        batch_preds = global_preds

                if self.num_outputs > 1:
                    batch_preds = F.softmax(cast(torch.Tensor, batch_preds), dim=-1)

            predictions.append(cast(torch.Tensor, batch_preds).numpy())
            cur_i += batch_size
            pbar.update(batch_size)

        all_preds = np.concatenate(predictions, axis=0)
        if len(all_preds.shape) == 1:
            all_preds = np.expand_dims(all_preds, axis=-1)

        if self.idx_to_output_class is not None:
            feature_labels: Optional[List[str]] = []
            for idx in range(len(self.idx_to_output_class)):
                cast(List, feature_labels).append(self.idx_to_output_class[idx])
        else:
            feature_labels = None
        return preds_to_xr(
            all_preds,
            lats=input_data.lat,
            lons=input_data.lon,
            feature_labels=feature_labels,
        )

    def get_interpretable_metrics(self, outputs, prefix: str) -> Dict:

        output_dict = {}

        # we want to calculate some more interpretable losses - accuracy,
        # and auc roc
        output_dict.update(
            self.single_output_metrics(
                torch.cat([x["pred"] for x in outputs]).detach().cpu().numpy(),
                torch.cat([x["label"] for x in outputs]).detach().cpu().numpy(),
                prefix=prefix,
            )
        )

        if self.hparams.multi_headed:
            output_dict.update(
                self.single_output_metrics(
                    torch.cat([x["Togo_pred"] for x in outputs]).detach().cpu().numpy(),
                    torch.cat([x["Togo_label"] for x in outputs])
                    .detach()
                    .cpu()
                    .numpy(),
                    prefix=f"{prefix}Togo_",
                )
            )
        return output_dict

    def single_output_metrics(
        self, preds: np.ndarray, labels: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:

        if len(preds) == 0:
            # sometimes this happens in the warmup
            return {}

        output_dict: Dict[str, float] = {}
        if not (labels == labels[0]).all():
            # This can happen when lightning does its warm up on a subset of the
            # validation data
            output_dict[f"{prefix}roc_auc_score"] = roc_auc_score(labels, preds)

        preds = (preds > self.hparams.probability_threshold).astype(int)

        output_dict[f"{prefix}precision_score"] = precision_score(labels, preds)
        output_dict[f"{prefix}recall_score"] = recall_score(labels, preds)
        output_dict[f"{prefix}f1_score"] = f1_score(labels, preds)
        output_dict[f"{prefix}accuracy"] = accuracy_score(labels, preds)

        return output_dict

    def _split_preds_and_get_loss(
        self, batch, add_preds: bool, loss_label: str, log_loss: bool
    ) -> Dict:

        x, label, is_togo = batch

        preds_dict: Dict = {}
        if self.hparams.multi_headed:
            global_preds, local_preds = self.forward(x)
            global_preds = global_preds[is_togo == 0]
            global_labels = label[is_togo == 0]

            local_preds = local_preds[is_togo == 1]
            local_labels = label[is_togo == 1]

            loss = 0
            if local_preds.shape[0] > 0:
                local_loss = self.loss_function(local_preds.squeeze(-1), local_labels)
                loss += local_loss

            if global_preds.shape[0] > 0:
                global_loss = self.loss_function(
                    global_preds.squeeze(-1), global_labels
                )
                num_local_labels = local_preds.shape[0]
                if num_local_labels == 0:
                    alpha = 1
                else:
                    ratio = global_preds.shape[0] / num_local_labels
                    alpha = ratio / self.hparams.alpha

                loss += alpha * global_loss
            if add_preds:
                preds_dict.update(
                    {
                        "pred": global_preds,
                        "label": global_labels,
                        "Togo_pred": local_preds,
                        "Togo_label": local_labels,
                    }
                )
        else:
            preds = self.forward(x)

            loss = self.loss_function(
                input=cast(torch.Tensor, preds).squeeze(-1), target=label,
            )

            if add_preds:
                preds_dict.update({"pred": preds, "label": label})

        output_dict: Dict = {loss_label: loss}
        if log_loss:
            output_dict["log"] = {loss_label: loss}
        output_dict.update(preds_dict)
        return output_dict

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # by default, this means no additional weighting will be given
        # if a region is passed, then we will assign a weighting of 10
        # (this happens in the dataloader, and empirically seems to work well. If
        # we do more experimenting with the hparams it might make sense to make it
        # modifiable here).
        parser_args: Dict[str, Tuple[Type, Any]] = {
            "--data_folder": (
                str,
                # this allows the model to be run from
                # anywhere on the machine
                str(Path("../data").absolute()),
            ),  # assumes this is being run from "scripts"
            "--model_base": (str, "lstm"),
            "--hidden_vector_size": (int, 64),
            "--learning_rate": (float, 0.001),
            "--batch_size": (int, 64),
            "--probability_threshold": (float, 0.5),
            "--num_classification_layers": (int, 2),
            "--alpha": (float, 10),
        }

        for key, val in parser_args.items():
            parser.add_argument(key, type=val[0], default=val[1])

        parser.add_argument("--add_togo", dest="add_togo", action="store_true")
        parser.add_argument("--exclude_togo", dest="add_togo", action="store_false")
        parser.set_defaults(add_togo=True)

        parser.add_argument("--add_geowiki", dest="add_geowiki", action="store_true")
        parser.add_argument(
            "--exclude_geowiki", dest="add_geowiki", action="store_false"
        )
        parser.set_defaults(add_geowiki=True)

        parser.add_argument(
            "--remove_b1_b10", dest="remove_b1_b10", action="store_true"
        )
        parser.add_argument("--keep_b1_b10", dest="remove_b1_b10", action="store_false")
        parser.set_defaults(remove_b1_b10=True)

        parser.add_argument("--multi_headed", dest="multi_headed", action="store_true")
        parser.add_argument(
            "--not_multi_headed", dest="multi_headed", action="store_false"
        )
        parser.set_defaults(multi_headed=True)

        temp_args = parser.parse_known_args()[0]
        return STR2BASE[temp_args.model_base].add_base_specific_arguments(parser)
