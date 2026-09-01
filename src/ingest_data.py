import os
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_DATA_DIR = PROJECT_ROOT / "extracted_data"


# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass


# Implement a concrete class for ZIP Ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Extracts a .zip file and returns the content as a pandas DataFrame."""

        # Ensure the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError(
                "The provided file is not a .zip file."
            )

        # Create extracted_data directory if it does not exist
        EXTRACTED_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Extract the ZIP file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(
                EXTRACTED_DATA_DIR
            )

        # Find CSV files inside extracted_data
        extracted_files = os.listdir(
            EXTRACTED_DATA_DIR
        )

        csv_files = [
            file
            for file in extracted_files
            if file.endswith(".csv")
        ]

        if len(csv_files) == 0:
            raise FileNotFoundError(
                "No CSV file found in the extracted data."
            )

        if len(csv_files) > 1:
            raise ValueError(
                "Multiple CSV files found. "
                "Please specify which one to use."
            )

        # Build full path to the extracted CSV
        csv_file_path = (
            EXTRACTED_DATA_DIR
            / csv_files[0]
        )

        # Read CSV into a DataFrame
        df = pd.read_csv(
            csv_file_path
        )

        return df


# Factory to create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(
        file_extension: str,
    ) -> DataIngestor:
        """Return the appropriate DataIngestor."""

        if file_extension == ".zip":
            return ZipDataIngestor()

        raise ValueError(
            f"No ingestor available for file extension: "
            f"{file_extension}"
        )


# Example usage
if __name__ == "__main__":

    file_path = (
        PROJECT_ROOT
        / "data"
        / "archive.zip"
    )

    data_ingestor = (
        DataIngestorFactory
        .get_data_ingestor(
            file_path.suffix
        )
    )

    df = data_ingestor.ingest(
        str(file_path)
    )

    print(df.head())