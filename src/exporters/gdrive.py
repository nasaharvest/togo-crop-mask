from pathlib import Path
import pickle

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import gdown

from .base import BaseExporter
from . import RegionalExporter

from typing import Dict, List, Optional


class GDriveExporter(BaseExporter):
    r"""
    An exporter to download data from Google Drive
    """

    dataset = "gdrive"  # we will only save the token here
    scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly"]

    def __init__(self, data_folder: Path = Path("data")) -> None:
        super().__init__(data_folder)

        assert (self.output_folder / "credentials.json").exists(), (
            f"Enable the google drive API at this link: "
            f"https://developers.google.com/drive/api/v3/quickstart/python "
            f"to use this class. Save the credentials.json at {self.output_folder}"
        )

        # https://developers.google.com/drive/api/v3/quickstart/python
        creds = None
        token_path = self.output_folder / "token.pickle"
        if token_path.exists():
            with token_path.open("rb") as f:
                creds = pickle.load(f)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.output_folder / "credentials.json", self.scopes
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with (self.output_folder / "token.pickle").open("wb") as token:
                pickle.dump(creds, token)

        self.service = build("drive", "v3", credentials=creds)

    def export(self, region_name: str, max_downloads: Optional[int] = None) -> None:
        r"""
        Download data from Google Drive. This is useful when downloading data exported by
        the regional exporter, as the filesizes can be large.

        :param region_name: The name of the downloaded region. The exporter will search for
            this string in the google drive files to filter which files to download
        :param max_downloads: The max number of downloads. If None, all tiff files containing
            region_name are downloaded
        """

        query = f'(fullText contains "{region_name}") and (mimeType = "image/tiff")'

        file_info: List[Dict] = []

        results = (
            self.service.files()
            .list(pageSize=10, q=query, fields="nextPageToken, files(id, name)",)
            .execute()
        )
        items = results.get("files", [])

        file_info.extend(items)

        next_page = results.get("nextPageToken", None)

        while next_page is not None:
            results = (
                self.service.files()
                .list(
                    pageSize=10,
                    pageToken=next_page,
                    # https://stackoverflow.com/questions/47402545/
                    # google-drive-js-api-nextpagetoken-invalid
                    q=query,
                    fields="nextPageToken, files(id, name)",
                )
                .execute()
            )

            items = results.get("files", [])
            file_info.extend(items)

            next_page = results.get("nextPageToken", None)

        print(f"Downloading {len(file_info)} files")

        for idx, individual_file in enumerate(file_info):
            if (max_downloads is not None) and (idx >= max_downloads):
                return None

            print(f"Downloading {individual_file['name']}")

            url = f"https://drive.google.com/uc?id={individual_file['id']}"

            download_path = (
                self.raw_folder / RegionalExporter.dataset / individual_file["name"]
            )
            if download_path.exists():
                print(f"File already exists! Skipping")
                continue

            gdown.download(url, str(download_path), quiet=False)
