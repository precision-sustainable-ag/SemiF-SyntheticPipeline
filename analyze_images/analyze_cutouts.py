from pathlib import Path
import cv2
import sys
import logging
import os
import random
from from_root import from_root
sys.path.append(str(from_root("src/utils")))
from sql3_query import SQLiteQueryHandler
sys.path.append(str(from_root("src")))
from create_recipes import DBRecipeManager, recursively_parse_json
from move_cutouts import CutoutDownloader
import uuid
import hydra
from hydra.utils import get_method
import shutil 

sys.path.append(str(from_root("analyze_images")))
# from graphs import save_batch_sizes_plot, save_shape_count_plot

log = logging.getLogger(__name__)

class DataCollector():
    def __init__(self, documents, cfg):
        self.documents = documents
        self.cfg = cfg

        self.unique_batches = set(doc['batch_id'].lower() for doc in self.documents)
        
        # Initialize the dictionary to store the image sizes for each image in a batch
        self.batch_image_dict = {}

        # Initialize the dictionary to store the number of components for each image in a batch
        self.batch_num_components = {}


    def get_data(self):

        print("Unique Vetch Batches")
        for name in sorted(self.unique_batches):
            print("-", name)

        # Iterate through the unique batch IDs
        for batch_id in sorted(self.unique_batches):
            # Filter documents for the current batch_id
            batch_docs = [doc for doc in self.documents if doc['batch_id'].lower() == batch_id]

            # Initialize the list for the current batch if it doesn't exist
            if batch_id not in self.batch_image_dict:
                self.batch_image_dict[batch_id] = []
            if batch_id not in self.batch_num_components:
                self.batch_num_components[batch_id] = []

            self.get_image_sizes(batch_docs, batch_id)
            self.get_num_components(batch_docs, batch_id)


        # save batch images for comparison
        self.save_images_based_on_batch()


    def get_image_sizes(self, batch_docs, batch_id):
        # Iterate through the filtered documents and calculate image sizes
        for doc in batch_docs:
            # Append the area (height × width) to the list corresponding to the batch_id
            self.batch_image_dict[batch_id].append((doc['cutout_height'], doc['cutout_width']))

    def get_num_components(self, batch_docs, batch_id):
        for doc in batch_docs:
            # Append the area (height × width) to the list corresponding to the batch_id
            self.batch_num_components[batch_id].append(doc['cutout_props']['num_components'])

    def save_images_based_on_size(self, img_name, img_size):
        os.makedirs(from_root('analyze_images/images/0-1000'   ), exist_ok=True)
        os.makedirs(from_root('analyze_images/images/1000-2000'), exist_ok=True)
        os.makedirs(from_root('analyze_images/images/2000-3000'), exist_ok=True)
        os.makedirs(from_root('analyze_images/images/3000-4000'), exist_ok=True)
        os.makedirs(from_root('analyze_images/images/4000-5000'), exist_ok=True)
        os.makedirs(from_root('analyze_images/images/5000-6000'), exist_ok=True)
        os.makedirs(from_root('analyze_images/images/6000-7000'), exist_ok=True)


    def save_images_based_on_batch(self):
        for batch_id in sorted(self.unique_batches):

            # Filter documents for the current batch_id
            batch_docs = [doc for doc in self.documents if doc['batch_id'].lower() == batch_id]

            # Collect unique lowercase cutout_ids
            unique_cutout_ids = list({doc['cutout_id'].lower() for doc in batch_docs})

            # # Sample up to 3 cutout_ids
            # num_samples = min(len(unique_cutout_ids))
            # sampled_cutout_ids = random.sample(unique_cutout_ids, num_samples)

            # Download sampled cutouts
            for cutout_id in unique_cutout_ids:
                self.download_image(cutout_id.upper(), batch_id.upper())
            print(f"Batch {batch_id} - Random cutout_ids: {unique_cutout_ids}")

    # taken directory from move_cutouts.py
    def download_image(self, cutout_id: str, batch_id: str):
        """
        Downloads an image corresponding to a given cutout_id and batch_id from the long-term storage.

        :param cutout_id: The ID of the cutout to download.
        :param batch_id: The batch ID to locate the cutout in the long-term storage.
        """
        image_filename = f"{cutout_id}.png"
        cutout_filename = f"{cutout_id}_mask.png"

        # Make batch directory
        os.makedirs(from_root(f'analyze_images/images/{batch_id}'), exist_ok=True)

        # List of storage locations (paths only)
        storages = [
            Path(self.cfg.paths.primary_longterm_storage, "semifield-cutouts"),
            Path(self.cfg.paths.secondary_longterm_storage, "semifield-cutouts"),
            Path(self.cfg.paths.tertiary_longterm_storage, "semifield-cutouts"),
        ]

        # Construct the local path where the image will be saved
        local_image_dir = from_root(f'analyze_images/images/{batch_id}')
        local_image_path = local_image_dir / image_filename
        local_cutout_path = local_image_dir / cutout_filename

        # Try each storage location until the image is found and copied
        for storage_path in storages:
            image_source_path = storage_path / batch_id / image_filename
            cutout_source_path = storage_path / batch_id / cutout_filename
            if image_source_path.exists():
                try:
                    shutil.copy(image_source_path, local_image_path)
                    log.debug(f"Downloaded: {image_filename} from {image_source_path} to {local_image_path}")
                    shutil.copy(cutout_source_path, local_cutout_path)
                    log.debug(f"Downloaded: {cutout_filename} from {image_source_path} to {local_image_path}")
                    return local_image_path  # Return path after successful download
                except IOError as e:
                    log.error(f"Error copying file from {image_source_path} to {local_image_path} - {e}")
                    continue  # Optionally, try next storage if copy fails



        # If we reach this point, the file was not found or could not be copied
        log.error(
            f"Image not found in any storage for cutout_id: {cutout_id}. Tried paths: " +
            ", ".join(str(storage_path / batch_id / image_filename) for storage_path in storages)
        )
        return None

class DataPreprocessor():
    def __init__(self):
        pass



# Function used to load data from sql database
def get_specified_data(cfg):
    query_handler = SQLiteQueryHandler(cfg)
    query_handler.add_conditions()
    rows, columns = query_handler.execute_query()
    query_handler.close()
    # Convert the rows to a list of dictionaries.
    documents = [dict(zip(columns, row)) for row in rows]
    for doc in documents:
        if "_id" not in doc:
            # Generate a new unique identifier as a string.
            doc["_id"] = str(uuid.uuid4())
    
    # Convert nested JSON strings into dictionaries/lists.
    documents = [recursively_parse_json(doc) for doc in documents]
    recipe_manager = DBRecipeManager(cfg)
    recipe_manager.process_cutouts(documents)

    return documents

@hydra.main(version_base="1.2", config_path=str(from_root("conf")), config_name="config")
def main(cfg):

    # make a directory to save images and graphs
    os.makedirs(from_root('analyze_images/images/'), exist_ok=True)
    os.makedirs(from_root('analyze_images/graphs/'), exist_ok=True)


    # load documents based on the config
    documents = get_specified_data(cfg)

    unique_common_names = set(doc['category']['common_name'].lower() for doc in documents)
    print(F"ANALYZING CUTOUTS FOR THE FOLLOWING {len(unique_common_names)} UNIQUE SPECIES: ")
    for name in sorted(unique_common_names):
        print("-", name)

    print("THE CONFIG SPECIFIES THE FOLLOWING DIRECTORIES")
    print(cfg.paths.primary_longterm_storage)
    print(cfg.paths.secondary_longterm_storage)
    print(cfg.paths.tertiary_longterm_storage)

    data = DataCollector(documents, cfg)

    data.get_data()
    # save_batch_sizes_plot(data.batch_image_dict)
    # save_shape_count_plot(data.batch_num_components)


if __name__ == '__main__':
    main()



