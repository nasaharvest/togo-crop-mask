import sys
from pathlib import Path
from datetime import date

sys.path.append("..")

from src.exporters import (
    GeoWikiExporter,
    GeoWikiSentinelExporter,
    RegionalExporter,
    TogoSentinelExporter,
    GDriveExporter,
    cancel_all_tasks,
)


def export_geowiki():

    exporter = GeoWikiExporter(Path("../data"))
    exporter.export()


def export_geowiki_sentinel_ee():
    exporter = GeoWikiSentinelExporter(Path("../data"))
    exporter.export_for_labels(num_labelled_points=None, monitor=False, checkpoint=True)


def export_togo():
    exporter = TogoSentinelExporter(Path("../data"))
    exporter.export_for_labels(
        num_labelled_points=None, monitor=False, checkpoint=True, evaluation_set=True
    )


def export_region():
    exporter = RegionalExporter(Path("../data"))
    exporter.export_for_region(
        region_name="Togo",
        end_date=date(2020, 4, 20),
        monitor=False,
        checkpoint=True,
        metres_per_polygon=None,
    )


def export_gdrive():
    exporter = GDriveExporter(Path("../data"))
    exporter.export(region_name="Togo", max_downloads=1)


def cancel_tasks():
    cancel_all_tasks()


if __name__ == "__main__":
    export_geowiki_sentinel_ee()
    export_togo()
    export_region()
    # cancel_all_tasks()
