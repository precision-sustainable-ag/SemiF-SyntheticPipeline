import json
import logging
import random
import uuid
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from pymongo import MongoClient
from omegaconf import DictConfig

from utils.query import MongoDBQueryHandler

log = logging.getLogger(__name__)
  
class RecipeCreator:
    """
    Class to create synthetic image recipes based on filtered cutouts.

    This class takes filtered cutouts (based on specific criteria) and 
    associates them with background images to create synthetic images.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the RecipeCreator class.

        Args:
            cfg (DictConfig): The configuration object that contains all necessary filter settings and paths.
        """
        self.cfg = cfg
        self.recipes = []  # Initialize an empty list to hold synthetic image recipes
        self.used_cutouts = set()  # Set to keep track of used cutouts

    def create_synthetic_image(self, background_image_id: str) -> Dict[str, Any]:
        """
        Create a new synthetic image entry with a unique synthetic_image_id.

        Args:
            background_image_id (str): The identifier of the background image.

        Returns:
            Dict[str, Any]: A new synthetic image dictionary.
        """
        synthetic_image_id = str(uuid.uuid4())  # Generate a unique ID for synthetic image
        
        # Create a new synthetic image entry with the background image ID and an empty list of cutouts
        synthetic_image = {
            "synthetic_image_id": synthetic_image_id,
            "background_image_id": background_image_id,
            "cutouts": []
        }
        self.recipes.append(synthetic_image)
        log.info(f"Created synthetic recipe with ID {synthetic_image_id}.")
        return synthetic_image

    def add_cutout_to_image(self, synthetic_image: Dict[str, Any], cutout: Dict[str, Any]) -> None:
        """
        Add a cutout to an existing synthetic image.

        Args:
            synthetic_image (Dict[str, Any]): The synthetic image to which the cutout will be added.
            cutout (Dict[str, Any]): The cutout metadata to be added.
        """
        # Ensure cutout ID is in string format for JSON compatibility
        if "_id" in cutout:
            cutout["_id"] = str(cutout["_id"])

        # Determine the maximum number of cutouts allowed for this synthetic image
        max_cutouts_per_image = random.randint(self.cfg.cutout_filters.cuts_n_image['min'], self.cfg.cutout_filters.cuts_n_image['max'])

        # Add cutout if the limit for the number of cutouts has not been reached
        if len(synthetic_image['cutouts']) < max_cutouts_per_image:
            synthetic_image['cutouts'].append(cutout)  # Add cutout to the image
            log.debug(f"Added cutout {cutout.get('cutout_id')} to synthetic image {synthetic_image['synthetic_image_id']}.")
        else:
            log.debug(f"Reached max cutouts for synthetic image {synthetic_image['synthetic_image_id']}.")

    def save_recipes(self, output_dir: str) -> None:
        """
        Save all generated synthetic image recipes to a JSON file.
        Args:
            output_dir (str): Directory to save the recipe files.
        """
        # Define the output file path
        recipe_filename = f"{self.cfg.general.project_name}_{self.cfg.general.sub_project_name}.json"
        output_path = Path(output_dir) / recipe_filename

        # Write recipes to a JSON file with proper formatting
        with open(output_path, 'w') as f:
            json.dump({"synthetic_images": self.recipes}, f, indent=4, default=str)
        log.info(f"Saved recipes to {output_path}.")


class MongoDBRecipeManager:
    """Main class to manage MongoDB document retrieval and recipe creation."""

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize MongoDB connection and prepare for processing cutouts.

        Args:
            cfg (DictConfig): The configuration object.
        """
        self.cfg = cfg
        # Set up MongoDB connection
        self.client = MongoClient(f'mongodb://{cfg.mongodb.host}:{cfg.mongodb.port}/')
        self.db = self.client[cfg.mongodb.db]  # Access MongoDB database
        self.collection = self.db[cfg.mongodb.collection]  # Access MongoDB collection
        self.recipe_creator = RecipeCreator(cfg)
        self.output_dir = Path(cfg.paths.projectdir, "recipes")  # Output directory for synthetic image recipes
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def weighted_sampling(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform weighted sampling based on the class weights defined in the config.

        Args:
            documents (List[Dict[str, Any]]): List of available cutouts.

        Returns:
            List[Dict[str, Any]]: List of sampled cutouts based on weights.
        """
        # Extract weights from config
        weights = self.cfg.cutout_filters.common_name_weights

        # Group documents by species/class
        class_groups = defaultdict(list)
        for doc in documents:
            # print(doc)
            # exit()
            class_groups[doc['category']['common_name']].append(doc)

        # Create a weighted population based on class weights
        weighted_population = []
        for class_name, group in class_groups.items():
            weight = weights.get(class_name, 1.0)  # Default weight of 1 if not in config
            weighted_population.extend(group * int(weight * 10))  # Adjust multiplier as needed

        return weighted_population

    def process_cutouts(self, documents: List[Dict[str, Any]]) -> None:
        """
        Process cutouts from MongoDB documents and create synthetic image recipes.

        Args:
            documents (List[Dict[str, Any]]): List of MongoDB documents containing cutout metadata.
        """
        # Gather all available background images (JPEG format) from the specified directory
        background_images = list(Path(self.cfg.paths.backgrounddir).glob('*.JPG')) + list(Path(self.cfg.paths.backgrounddir).glob('*.jpg'))

        # Total number of synthetic images to generate
        total_images = self.cfg.cutout_filters.total_images
        # Calculate the total number of cutouts needed based on configuration
        required_cutouts = total_images * self.cfg.cutout_filters.cuts_n_image['max']

        # Check if there are enough documents to meet the cutout requirements
        if len(documents) < required_cutouts and not self.cfg.cutout_filters.reuse_cutouts:
            log.warning("Not enough documents to fulfill the required cutouts without reuse.")

        # Use weighted sampling to create a pool of cutouts
        if self.cfg.cutout_filters.common_name_weights:
            available_cutouts = self.weighted_sampling(documents)
        else:
            available_cutouts = documents.copy()  # Copy documents to avoid modifying the original list

        for image_index in range(total_images):
            # Stop processing if no more available cutouts
            if not available_cutouts:
                log.warning("No more available cutouts to process.")
                break

            # Randomly select a background image from the available options
            background_image_id_path = random.choice(background_images)
            background_image_id = background_image_id_path.name

            # Determine the number of cutouts for this synthetic image
            num_cutouts_for_image = random.randint(self.cfg.cutout_filters.cuts_n_image['min'], self.cfg.cutout_filters.cuts_n_image['max'])

            # If there are not enough cutouts left and reuse is not allowed, log a warning
            if len(available_cutouts) < num_cutouts_for_image and not self.cfg.cutout_filters.reuse_cutouts:
                log.warning(f"Not enough cutouts for image {image_index + 1}, using {len(available_cutouts)} instead.")
                sampled_cutouts = available_cutouts  # Use all available cutouts
            elif len(available_cutouts) < num_cutouts_for_image:
                log.warning(f"Not enough cutouts for image {image_index + 1}, reusing cutouts.")
                sampled_cutouts = available_cutouts  # Use all available cutouts
                sampled_cutouts = random.choices(available_cutouts, k=num_cutouts_for_image)
            else:
                sampled_cutouts = random.sample(available_cutouts, num_cutouts_for_image)  # Randomly sample the required number of cutouts

            cutouts_added_to_image = False  # Flag to check if cutouts were successfully added

            # Create a synthetic image with the selected background image
            synthetic_image = self.recipe_creator.create_synthetic_image(background_image_id)
            
            # Add each sampled cutout to the synthetic image
            for cutout in sampled_cutouts:
                if not self.cfg.cutout_filters.reuse_cutouts and cutout["_id"] in self.recipe_creator.used_cutouts:
                    continue  # Skip cutout if it has already been used and reuse is not allowed
                self.recipe_creator.used_cutouts.add(cutout["_id"])  # Track used cutouts
                self.recipe_creator.add_cutout_to_image(synthetic_image, cutout)  # Add cutout to the synthetic image
                cutouts_added_to_image = True  # Set flag to True

            # Remove used cutouts from available cutouts if reuse is not allowed
            if not self.cfg.cutout_filters.reuse_cutouts:
                available_cutouts = [doc for doc in available_cutouts if doc["_id"] not in self.recipe_creator.used_cutouts]

            # Log the creation of the synthetic image if any cutouts were added
            if cutouts_added_to_image:
                log.info(f"Generated {image_index + 1} / {total_images} images.")
            else:
                log.warning(f"No cutouts added for image {image_index + 1}. Skipping.")

        # Save all the generated recipes to JSON
        self.recipe_creator.save_recipes(self.output_dir)
        log.info(f"Total images generated: {min(image_index + 1, total_images)}")


def main(cfg: DictConfig) -> None:
    """
    Main function to initialize the MongoDBRecipeManager and start the recipe creation process.
    """
    log.info("Starting recipe creation process.")  # Log the start of the process
    query_handler = MongoDBQueryHandler(cfg)
    query_handler.build_query()
    documents = query_handler.execute_query()

    recipe_manager = MongoDBRecipeManager(cfg)
    recipe_manager.process_cutouts(documents)
    log.info("Recipe creation completed.")
