import logging
import shutil
import os
from typing import List
from pathlib import Path
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from datetime import datetime

log = logging.getLogger(__name__)

class CutoutDownloader:
    def __init__(self, nfs_base_dir: str, local_download_dir: str):
        """
        Initializes the CutoutDownloader object.

        :param nfs_base_dir: Base directory on the NFS where the batch folders are stored.
        :param local_download_dir: Local directory to store copied cutouts.
        """
        self.nfs_base_dir = Path(nfs_base_dir)
        self.local_download_dir = Path(local_download_dir)
        self.local_download_dir.mkdir(parents=True, exist_ok=True)
        self.report_data = {}
        log.info(f"Initialized CutoutDownloader with NFS base directory: {self.nfs_base_dir} and local download directory: {self.local_download_dir}")

    def _get_folder_size(self, folder: Path) -> int:
        """
        Calculates the total size of a folder in bytes.

        :param folder: The Path object of the folder.
        :return: The size of the folder in bytes.
        """
        total_size = 0
        for path, dirs, files in os.walk(folder):
            for file in files:
                fp = Path(path) / file
                total_size += fp.stat().st_size
        log.debug(f"Calculated size for folder {folder}: {total_size} bytes")
        return total_size

    def download_batch(self, batch_name: str) -> None:
        """
        Copies all cutouts within a given batch from the NFS to the local directory.
        Skips the download if the batch is already fully downloaded.

        :param batch_name: The name of the batch to copy.
        """
        nfs_batch_dir = self.nfs_base_dir / batch_name
        local_batch_dir = self.local_download_dir / batch_name

        # Get the list of expected files in the source directory
        expected_cutouts = self._get_cutouts_list(nfs_batch_dir)
        expected_files_count = len(expected_cutouts) * 4  # Each cutout has 4 associated files

        # Check if the batch is already fully downloaded
        if local_batch_dir.exists():
            local_files_count = len(list(local_batch_dir.glob('*')))
            if local_files_count == expected_files_count:
                log.info(f"Batch '{batch_name}' is already fully downloaded. Skipping.")
                return
        else:
            local_batch_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Created directory for batch '{batch_name}' at {local_batch_dir}")

        # Proceed with download if not fully downloaded
        log.info(f"Starting download for batch: {batch_name}")
        max_workers = int(len(os.sched_getaffinity(0)) / 5)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._copy_cutout_files, nfs_batch_dir, local_batch_dir, cutout)
                       for cutout in expected_cutouts]
            for future in as_completed(futures):
                future.result()

        # Update report data
        self.report_data[batch_name] = {
            'num_cutouts': len(expected_cutouts),
            'folder_size_mib': self._get_folder_size(local_batch_dir) / (1024 * 1024),  # Convert bytes to MiB
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        log.info(f"Completed download for batch '{batch_name}' with {len(expected_cutouts)} cutouts.")

    def _get_cutouts_list(self, nfs_batch_dir: Path) -> List[str]:
        """
        Retrieves the list of cutouts in a batch, ensuring only relevant files are counted.

        :param nfs_batch_dir: Path to the batch directory on the NFS.
        :return: A list of cutout file stems (without extensions).
        """
        # List all files in the directory and exclude any CSV files
        cutout_files = [f for f in nfs_batch_dir.glob('*') if f.is_file() and not f.suffix == '.csv']
        cutouts = set(file.stem.split('_mask')[0] for file in cutout_files)
        log.debug(f"Found {len(cutouts)} unique cutouts in batch directory {nfs_batch_dir}")
        return list(cutouts)

    def _copy_cutout_files(self, nfs_batch_dir: Path, local_batch_dir: Path, cutout_stem: str) -> None:
        """
        Copies the files associated with a single cutout from the NFS to the local directory.

        :param nfs_batch_dir: Path to the batch directory on the NFS.
        :param local_batch_dir: Local directory to store the cutout files.
        :param cutout_stem: The file stem for the cutout.
        """
        extensions = ['.jpg', '.png', '_mask.png', '.json']
        for ext in extensions:
            file_name = f"{cutout_stem}{ext}"
            nfs_file_path = nfs_batch_dir / file_name
            local_file_path = local_batch_dir / file_name
            
            if nfs_file_path.exists():
                shutil.copy2(nfs_file_path, local_file_path)
                log.debug(f"Copied {nfs_file_path} to {local_file_path}")

    def download_batches(self, batch_list: List[str]) -> None:
        """
        Copies all batches specified in the batch list from the NFS to the local directory.

        :param batch_list: A list of batch names to copy.
        """
        for batch in batch_list:
            log.info(f"Processing batch: {batch}")
            self.download_batch(batch)

    def generate_report(self, report_file: str) -> None:
        """
        Generates a CSV report of the download process, including all batches present in the local download directory.

        :param report_file: Path to the report CSV file.
        """
        report_path = Path(report_file)

        # Check if the report file exists
        if report_path.exists():
            # Load the existing report
            existing_df = pd.read_csv(report_path)
            log.info(f"Loaded existing report from {report_path}")
        else:
            # Create an empty DataFrame if the report does not exist
            existing_df = pd.DataFrame(columns=['Batch Name', 'Number of Unique Cutouts', 'Folder Size (MiB)', 'Timestamp'])
            log.info(f"Creating a new report file at {report_path}")

        # List of currently present batch names
        present_batches = []

        # Scan the local download directory for all processed batches
        for batch_dir in self.local_download_dir.iterdir():
            if batch_dir.is_dir():  # Only process directories
                batch_name = batch_dir.name
                present_batches.append(batch_name)
                cutouts = self._get_cutouts_list(batch_dir)
                self.report_data[batch_name] = {
                    'num_cutouts': len(cutouts),
                    'folder_size_mib': self._get_folder_size(batch_dir) / (1024 * 1024),  # Convert bytes to MiB
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

        # Create a DataFrame from the current report data
        new_data = {
            'Batch Name': [],
            'Number of Unique Cutouts': [],
            'Folder Size (MiB)': [],
            'Timestamp': []
        }

        for batch_name, data in self.report_data.items():
            new_data['Batch Name'].append(batch_name)
            new_data['Number of Unique Cutouts'].append(data['num_cutouts'])
            new_data['Folder Size (MiB)'].append(f"{data['folder_size_mib']:.2f}")
            new_data['Timestamp'].append(data['timestamp'])

        new_df = pd.DataFrame(new_data)

        # Combine existing data with new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Remove duplicates in case the same batch is processed again and filter only present batches
        combined_df.drop_duplicates(subset=['Batch Name'], keep='last', inplace=True)
        combined_df = combined_df[combined_df['Batch Name'].isin(present_batches)]
        combined_df = combined_df.sort_values(by='Batch Name')

        # Save the filtered DataFrame back to the CSV file
        combined_df.to_csv(report_path, index=False)
        log.info(f"Report generated and saved to {report_path}")

    @staticmethod
    def load_batch_list(file_path: str) -> List[str]:
        """
        Loads the batch list from a text file.

        :param file_path: Path to the text file containing batch names.
        :return: A list of batch names.
        """
        with open(file_path, 'r') as file:
            batch_list = [line.strip() for line in file.readlines()]
        log.info(f"Loaded batch list from {file_path} with {len(batch_list)} batches")
        return batch_list


def main(cfg: DictConfig) -> None:
    # Base directory on the NFS where the batch folders are located
    nfs_base_dir = Path(cfg.paths.longterm_storage)
    
    # Local directory to save the copied cutouts
    local_download_dir = Path(cfg.paths.datadir, "cutouts")
    
    # Path to the text file containing the list of batches to copy
    batch_list_file = Path(cfg.paths.datadir, "batch_list.txt")

    # Path to the report CSV file
    report_file = Path(cfg.paths.datadir, "cutout_report.csv")

    # Load the batch list from the text file
    batch_list = CutoutDownloader.load_batch_list(batch_list_file)

    # Initialize the downloader and copy the batches with threading
    downloader = CutoutDownloader(nfs_base_dir, local_download_dir)
    downloader.download_batches(batch_list)

    # Generate the report and update the CSV file
    downloader.generate_report(report_file)
    log.info(f"Report generated and updated: {report_file}")
