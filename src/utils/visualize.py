import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from pathlib import Path

from sklearn.metrics import roc_curve, auc

from typing import Optional


def sentinel_as_tci(sentinel_ds: xr.DataArray, scale: bool = True) -> xr.DataArray:
    r"""
    Get a True Colour Image from Sentinel data exported from Earth Engine
    :param sentinel_ds: The sentinel data, exported from Earth Engine
    :param scale: Whether or not to add the factor 10,000 scale
    :return: A dataframe with true colour bands
    """

    band2idx = {
        band: idx for idx, band in enumerate(sentinel_ds.attrs["band_descriptions"])
    }

    tci_bands = ["B4", "B3", "B2"]
    tci_indices = [band2idx[band] for band in tci_bands]
    if scale:
        return sentinel_ds.isel(band=tci_indices) / 10000 * 2.5
    else:
        return sentinel_ds.isel(band=tci_indices) * 2.5


def plot_roc_curve(
    true: np.ndarray, preds: np.ndarray, threshold_spacing: Optional[int] = 3
) -> None:
    fpr, tpr, thresholds = roc_curve(true, preds)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(fpr, tpr, label="AUC ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"AUC ROC: {auc(fpr, tpr)}")
    ax.legend(loc="lower right")

    if threshold_spacing is not None:

        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)

        for idx, thresh in enumerate(thresholds):
            if (idx + 1) % threshold_spacing == 0:
                plt.scatter(fpr[idx], tpr[idx])
                plt.annotate(
                    round(thresh, 2),
                    (fpr[idx], tpr[idx]),
                    textcoords="offset points",
                    xytext=(-20, 10),  # distance from text to points (x,y)
                    bbox=bbox_props,
                )
    plt.show()


def plot_with_mask(
    path_to_org: Path, path_to_preds: Path, threshold: float = 0.5
) -> None:
    """
    The ordering of the dimensions might be funky
    """
    tci = sentinel_as_tci(xr.open_dataset(path_to_org).FEATURES, scale=False).isel(
        time=-1
    )

    preds = xr.open_dataset(path_to_preds)

    tci = tci.sortby("x").sortby("y")
    # This feels hacky? but it works
    preds = preds.transpose("lat", "lon").sortby("lat", ascending=False).sortby("lon")

    plt.clf()
    fig, ax = plt.subplots(
        1, 3, figsize=(20, 7.5), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    fig.suptitle(
        f"Model results for tile with bottom left corner:"
        f"\nat latitude {float(preds.lat.min())}"
        f"\n and longitude {float(preds.lon.min())}",
        fontsize=15,
    )
    # ax 1 - original
    img_extent_1 = (tci.x.min(), tci.x.max(), tci.y.min(), tci.y.max())
    img = np.clip(np.moveaxis(tci.values, 0, -1), 0, 1)

    ax[0].set_title("True colour image")
    ax[0].imshow(img, origin="upper", extent=img_extent_1, transform=ccrs.PlateCarree())

    kwargs = {
        "extent": img_extent_1,
        "transform": ccrs.PlateCarree(),
    }

    if len(preds.data_vars) == 1:
        raw_preds = preds.prediction_0
        mask = preds.prediction_0 > threshold

        kwargs.update({"vmin": 0, "vmax": 1})
    else:
        raw_preds = mask = np.argmax(preds.to_array().values, axis=0)

    # ax 2 - mask
    ax[1].set_title("Mask")

    im = ax[1].imshow(raw_preds, **kwargs)

    # finally, all together
    ax[2].set_title("Mask on top of the true colour image")
    ax[2].imshow(img, origin="upper", extent=img_extent_1, transform=ccrs.PlateCarree())

    kwargs["alpha"] = 0.3
    ax[2].imshow(mask, **kwargs)

    if len(preds.data_vars) == 1:
        fig.colorbar(
            im, ax=ax.ravel().tolist(),
        )

    else:
        # This function formatter will replace integers with target names
        formatter = plt.FuncFormatter(lambda val, loc: list(preds.data_vars)[val])

        # We must be sure to specify the ticks matching our target names
        fig.colorbar(
            im,
            ticks=range(len(preds.data_vars)),
            format=formatter,
            ax=ax.ravel().tolist(),
        )
