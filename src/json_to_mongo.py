import json
from pymongo import MongoClient
from pathlib import Path
from typing import Union, List
from omegaconf import DictConfig
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

class MongoDBDataLoader:
    """Class to handle loading JSON data into MongoDB from an NFS storage locker based on batches."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize MongoDB connection.

        Args:
            cfg (DictConfig): OmegaConf DictConfig object containing the configuration settings.
        """
        self.cfg = cfg
        self.client = MongoClient(f'mongodb://{cfg.mongodb.host}:{cfg.mongodb.port}/')
        self.db = self.client[cfg.mongodb.db]
        self.collection = self.db[cfg.mongodb.collection]
        
        # Root directory of the NFS storage locker
        self.primary_nfs_root = Path(cfg.paths.primary_longterm_storage)
        self.secondary_nfs_root = Path(cfg.paths.secondary_longterm_storage)
     

    def load_batches_from_yaml(self) -> List[str]:
        """
        Load batch names from a YAML configuration file.

        Args:
            yaml_file_path (Union[str, Path]): Path to the YAML file.
        """
        batches = self.cfg.batches
        log.info(f"Loaded {len(batches)} batches from YAML file.")
        return batches

    def load_json_files_from_batches(self, batches: List[str]) -> None:
        """
        Load JSON files from batch directories in the NFS storage locker and insert them into MongoDB.
        If the batch is not found in the primary storage path, the script checks the alternative storage path.

        Args:
            batches (List[str]): List of batch directories to process.
        """
        primary_nfs_root_path = Path(self.primary_nfs_root)
        secondary_nfs_root_path = Path(self.secondary_nfs_root)

        # Loop through all the batch directories
        for batch_name in tqdm(batches):
            batch_dir = primary_nfs_root_path / batch_name

            # Check if the batch exists in the primary storage
            if batch_dir.is_dir():
                json_files = list(batch_dir.glob('*.json'))
                log.info(f"Processing batch '{batch_name}' in primary storage with {len(json_files)} JSON files.")
            else:
                # If not found, check the alternative storage
                batch_dir = secondary_nfs_root_path / batch_name
                if batch_dir.is_dir():
                    json_files = list(batch_dir.glob('*.json'))
                    log.info(f"Processing batch '{batch_name}' in alternative storage with {len(json_files)} JSON files.")
                else:
                    log.warning(f"Batch directory '{batch_name}' not found in either primary or alternative storage.")
                    continue

            for json_file_path in json_files:
                self.insert_data_from_file(json_file_path)


    def insert_data_from_file(self, json_file_path: Union[str, Path]) -> None:
        """
        Insert data from a JSON file into the MongoDB collection.

        Args:
            json_file_path (Union[str, Path]): Path to the JSON file.
        """
        try:
            with open(json_file_path) as file:
                data = json.load(file)

            # Check if the data is a list or a dictionary and insert accordingly
            if isinstance(data, list):
                self.collection.insert_many(data, bypass_document_validation=False)
            else:
                self.collection.insert_one(data, bypass_document_validation=False)

            # log.info(f"Data from {json_file_path} inserted successfully!")

        except Exception as e:
            log.error(f"Failed to insert data from {json_file_path}: {e}")


# Example usage:
def main(cfg: DictConfig) -> None:
    # Set up the loader with MongoDB connection
    data_loader = MongoDBDataLoader(cfg)

    # Load batch names from the YAML file
    batch_names = data_loader.load_batches_from_yaml()

    # Load and insert JSON files from the specified batch directories
    data_loader.load_json_files_from_batches(batch_names)
