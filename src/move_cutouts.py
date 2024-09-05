import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict

from omegaconf import DictConfig

from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)


class CutoutDownloader:
    """
    Class to handle downloading cutout images from long-term storage to a local directory.
    """

    def __init__(self, cfg:DictConfig) -> None:
        """
        Initializes the CutoutDownloader with paths for JSON data, long-term storage, and local folder.

        :param json_file_path: Path to the JSON file containing cutout metadata.
        :param longterm_storage_base: Base path for the long-term storage where the images are stored.
        :param local_download_folder: Local folder where the images will be downloaded.
        """
        self.json_file_path = Path(cfg.paths.projectdir,"recipes", f"{cfg.general.project_name}_{cfg.general.sub_project_name}.json")
        
        self.primary_storage_base = Path(cfg.paths.primary_longterm_storage)
        self.secondary_storage_base = Path(cfg.paths.secondary_longterm_storage)
        self.local_download_folder = Path(cfg.paths.cutoutdir)
        self.max_workers = cfg.move_cutouts.parallel_workers


        # Create the local download folder if it doesn't exist
        if not self.local_download_folder.exists():
            self.local_download_folder.mkdir(parents=True, exist_ok=True)
            log.info(f"Created local download folder: {self.local_download_folder}")

    def load_json(self) -> List[Dict]:
        """
        Loads the JSON data from the specified file.

        :return: List of synthetic image dictionaries containing cutout information.
        """
        try:
            with open(self.json_file_path, "r") as f:
                data = json.load(f)
                log.info(f"Successfully loaded JSON data from {self.json_file_path}")
                return data["synthetic_images"]
        except FileNotFoundError as e:
            log.error(f"JSON file not found: {self.json_file_path} - {e}")
            raise
        except json.JSONDecodeError as e:
            log.error(f"Error decoding JSON file: {self.json_file_path} - {e}")
            raise

    def download_image(self, cutout_id: str, batch_id: str) -> None:
        """
        Downloads an image corresponding to a given cutout_id and batch_id from the long-term storage.

        :param cutout_id: The ID of the cutout to download.
        :param batch_id: The batch ID to locate the cutout in the long-term storage.
        """
        # Construct the file path in long-term storage
        image_filename = f"{cutout_id}.png"
        
        primary_image_path = Path(self.primary_storage_base, batch_id, image_filename)
        secondary_image_path = Path(self.secondary_storage_base, batch_id, image_filename)
        
        # Construct the local path to save the image
        local_image_path = Path(self.local_download_folder, image_filename)

        # Check if the file exists in the primary storage
        if primary_image_path.exists():
            try:
                shutil.copy(primary_image_path, local_image_path)
                log.debug(f"Downloaded from primary: {cutout_id} to {local_image_path}")
            except IOError as e:
                log.error(f"Error copying file from primary: {primary_image_path} to {local_image_path} - {e}")
        
        # If the file does not exist in primary, try the secondary storage
        elif secondary_image_path.exists():
            try:
                shutil.copy(secondary_image_path, local_image_path)
                log.debug(f"Downloaded from secondary: {cutout_id} to {local_image_path}")
            except IOError as e:
                log.error(f"Error copying file from secondary: {secondary_image_path} to {local_image_path} - {e}")
        
        else:
            log.error(f"Image not found in both primary and secondary storage: {cutout_id}")


    def get_unique_cutouts(self, synthetic_images: List[Dict]) -> Dict[str, str]:
        """
        Extracts unique cutout_ids from the list of synthetic images and their associated batch_ids.

        :param synthetic_images: List of dictionaries representing synthetic images.
        :return: Dictionary with unique cutout_ids as keys and their corresponding batch_ids as values.
        """
        unique_cutouts = {}
        for synthetic_image in synthetic_images:
            for cutout in synthetic_image["cutouts"]:
                cutout_id = cutout["cutout_id"]
                batch_id = cutout["batch_id"]
                if cutout_id not in unique_cutouts:
                    unique_cutouts[cutout_id] = batch_id
        return unique_cutouts
    
    def process_cutouts_sequentially(self) -> None:
        """
        Processes each cutout in the JSON file and downloads the corresponding images from long-term storage in serial mode.
        """
        synthetic_images = self.load_json()

        unique_cutouts = self.get_unique_cutouts(synthetic_images)
        log.info(f"Found {len(unique_cutouts)} unique cutouts to download.")
        
        for cutout_id, batch_id in unique_cutouts.items():
            self.download_image(cutout_id, batch_id)
        
        log.info("Download process completed in serial mode.")

    def process_cutouts_concurrently(self) -> None:
        """
        Processes each cutout in the JSON file and downloads the corresponding images from long-term storage.
        This method uses multithreading to parallelize the download process.
        """
        synthetic_images = self.load_json()

        # List of tasks to be submitted to the thread pool
        tasks = []

        # Thread pool executor for multithreading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for synthetic_image in synthetic_images:
                for cutout in synthetic_image["cutouts"]:
                    cutout_id = cutout["cutout_id"]
                    batch_id = cutout["batch_id"]
                    # Submit a task to download each image
                    tasks.append(executor.submit(self.download_image, cutout_id, batch_id))

            # Wait for all tasks to complete
            for future in as_completed(tasks):
                try:
                    future.result()  # Get the result (this will re-raise exceptions if any occurred)
                except Exception as e:
                    log.error(f"Error occurred during download task: {e}")

        log.info("Download process completed.")



def main(cfg: DictConfig) -> None:
    
    # Create an instance of CutoutDownloader and run the process
    downloader = CutoutDownloader(cfg)

    if cfg.move_cutouts.parallel:
        downloader.process_cutouts_concurrently()
    else:
        downloader.process_cutouts_sequentially()
