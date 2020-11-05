# Togo Crop Mask

A pixel-wise land type classifier, used to generate a crop mask for Togo

## Introduction

This repository contains code and data to generate a crop mask for Togo.
It was used to deliver a high-resolution (10m) cropland mask in 10 days to help the government distribute aid to smallholder farmers during the COVID-19 pandemic.

<img src="diagrams/results.jpg" alt="model results" height="400px"/>

It combines a hand-labelled dataset of crop / non-crop images with a [global database of crowdsourced cropland data](https://doi.pangaea.de/10.1594/PANGAEA.873912)
to train a multi-headed LSTM-based model to predict the presence of cropland in a pixel.

The map can be found on [Google Earth Engine](https://code.earthengine.google.com/5d8ff282e63c26610b7cd3b4a989929c).

## Pipeline

The main entrypoints into the pipeline are the [scripts](scripts). Specifically:

* [scripts/export.py](scripts/export.py) exports data (locally, or to Google Drive - see below)
* [scripts/process.py](scripts/process.py) processes the raw data
* [scripts/engineer.py](scripts/engineer.py) combines the earth observation data with the labels to create (x, y) training data
* [scripts/models.py](scripts/models.py) trains the models
* [scripts/predict.py](scripts/predict.py) takes a trained model and runs it on an area, generating plots like the one above

The [split_tiff.py](scripts/split_tiff.py) script is useful to break large exports from Google Earth Engine, which may
be too large to fit into memory.

## Setup

[Anaconda](https://www.anaconda.com/download/#macos) running python 3.6 is used as the package manager. To get set up
with an environment, install Anaconda from the link above, and (from this directory) run

```bash
conda env create -f environment.yml
```
This will create an environment named `landcover-mapping` with all the necessary packages to run the code. To
activate this environment, run

```bash
conda activate landcover-mapping
```

#### Earth Engine

Earth engine is used to export data. To use it, once the conda environment has been activated, run

```bash
earthengine authenticate
```

and follow the instructions. To test that everything has worked, run

```bash
python -c "import ee; ee.Initialize()"
```

Note that Earth Engine exports files to Google Drive by default (to the same google account used sign up to Earth Engine).

Running exports can be viewed (and individually cancelled) in the `Tabs` bar on the [Earth Engine Code Editor](https://code.earthengine.google.com/).
For additional support the [Google Earth Engine forum](https://groups.google.com/forum/#!forum/google-earth-engine-developers) is super
helpful.

Exports from Google Drive should be saved in [`data/raw`](data/raw).
This happens by default if the [GDrive](src/exporters/gdrive.py) exporter is used.

#### Tests

The following tests can be run against the pipeline:

```bash
pytest  # unit tests, written in the test folder
black .  # code formatting
mypy src  # type checking
```

## Reference

If you find this code useful, please cite the following paper:

Hannah Kerner, Gabriel Tseng, Inbal Becker-Reshef, Catherine Nakalembe, Brian Barker, Blake Munshell, Madhava Paliyam, and Mehdi Hosseini. 2020. Rapid Response Crop Maps in Data Sparse Regions. In review for KDD ’20: ACMSIGKDD Conference on Knowledge Discovery and Data Mining Workshops, August 22–27, 2020, San Diego, CA.

The hand-labeled training and test data used in the above paper can be found at: https://doi.org/10.5281/zenodo.3836629
