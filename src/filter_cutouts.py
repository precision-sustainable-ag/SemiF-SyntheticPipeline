import json
import os
from pathlib import Path
from typing import Dict, List, Any
import logging
from omegaconf import DictConfig
import random

# Set up logging
log = logging.getLogger(__name__)

class CutoutFilter:
    """Class to filter cutouts based on metadata criteria defined in a configuration file."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the CutoutFilter with the given configuration.

        :param cfg: Configuration object containing filtering criteria.
        """
        self.cfg = cfg

    def _check_range(self, value: float, min_val: float, max_val: float) -> bool:
        """Check if a value falls within the specified range."""
        return min_val <= value <= max_val

    def _check_exact(self, value: Any, target: Any) -> bool:
        """Check if a value exactly matches the target."""
        return value == target

    def _check_list(self, value: Any, target_list: List[Any]) -> bool:
        """Check if a value is within the target list."""
        return value in target_list

    def filter_cutout(self, cutout_metadata: Dict[str, Any]) -> bool:
        """
        Filter a single cutout based on the configuration criteria.

        :param cutout_metadata: Dictionary containing the cutout metadata.
        :return: True if the cutout matches all filtering criteria, False otherwise.
        """
        # Check morphological properties
        morph_props = cutout_metadata.get("cutout_props", {})
        morph_config = self.cfg.cutout_filters.morphological

        for prop, settings in morph_config.items():
            if isinstance(settings, dict):  # Range-based settings
                if not self._check_range(morph_props.get(prop, 0), settings['min'], settings['max']):
                    return False
            elif isinstance(settings, bool):  # Boolean settings
                if morph_props.get(prop) != settings:
                    return False

        # Check category properties
        category = cutout_metadata.get("category", {})
        category_config = self.cfg.cutout_filters.category

        for prop, target in category_config.items():
            if target and not self._check_exact(category.get(prop), target):
                return False

        return True


class RecipeCreator:
    """Class to create synthetic image recipes based on filtered cutouts."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the RecipeCreator with the given filter configuration.

        :param cfg: Configuration object containing filtering criteria.
        """
        self.filter = CutoutFilter(cfg)
        self.cfg = cfg
        self.recipes = []

    def add_cutout_to_image(self, image_id: str, cutout: Dict[str, Any]) -> None:
        """
        Add a filtered cutout to a synthetic image.

        :param image_id: Background image ID to associate the cutout with.
        :param cutout: Metadata of the cutout to be added.
        """
        # Find or create the synthetic image entry
        synthetic_image = next((img for img in self.recipes if img['background_image_id'] == image_id), None)
        
        if synthetic_image is None:
            synthetic_image = {
                "background_image_id": image_id,
                "cutouts": []
            }
            self.recipes.append(synthetic_image)

        # Check if the number of cutouts per image has been reached
        if len(synthetic_image['cutouts']) < self.cfg.cutout_filters.cuts_n_image['max']:
            synthetic_image['cutouts'].append(cutout)
            log.info(f"Cutout {cutout.get('cutout_id')} added to synthetic image {image_id}.")

    def save_recipes(self, output_path: str) -> None:
        """
        Save the created recipes to a JSON file.

        :param output_path: File path where the recipe JSON will be saved.
        """
        with open(output_path, 'w') as f:
            json.dump({"synthetic_images": self.recipes}, f, indent=4)
        log.info(f"Recipes saved to {output_path}")


class CutoutManager:
    """Main class to manage the cutout filtering and recipe creation process."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the CutoutManager with the paths to the configuration, cutouts, and output directory.

        :param cfg: Configuration object containing filtering criteria and paths.
        """
        self.cfg = cfg
        self.cutouts_dir = cfg.paths.cutoutdir    
        self.output_dir = Path(cfg.paths.recipesdir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recipe_creator = RecipeCreator(self.cfg)

    def process_cutouts(self) -> None:
        """
        Process each cutout one by one, applying the filtering criteria and adding matching cutouts to the recipe.
        """
        background_images = self.cfg.paths.backgrounddir
        min_cutouts_per_image = self.cfg.cutout_filters.cuts_n_image['min']
        max_cutouts_per_image = self.cfg.cutout_filters.cuts_n_image['max']

        for filename in os.listdir(self.cutouts_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.cutouts_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        metadata = json.load(f)
                        log.info(f"Processing metadata from {filename}")

                        if self.recipe_creator.filter.filter_cutout(metadata):
                            # Randomly assign the cutout to a background image
                            background_image_id = random.choice(background_images)
                            self.recipe_creator.add_cutout_to_image(background_image_id, metadata)

                            # Check if we need to stop processing to respect the cuts_n_image limits
                            for synthetic_image in self.recipe_creator.recipes:
                                if len(synthetic_image['cutouts']) < min_cutouts_per_image:
                                    break
                            else:
                                log.info("Desired number of cutouts per image reached.")
                                break

                except Exception as e:
                    log.error(f"Error processing file {filename}: {e}")

        # Save the recipes to the output directory
        recipe_filename = f"recipe_{len(os.listdir(self.output_dir)) + 1}.json"
        output_path = os.path.join(self.output_dir, recipe_filename)
        self.recipe_creator.save_recipes(output_path)


def main(cfg: DictConfig) -> None:
    log.info("Starting synthetic image generation.")
    cutout_manager = CutoutManager(cfg)
    cutout_manager.process_cutouts()
    log.info("Synthetic image generation completed.")
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import logging
from omegaconf import DictConfig
import random

# Set up logging
log = logging.getLogger(__name__)

class CutoutFilter:
    """Class to filter cutouts based on metadata criteria defined in a configuration file."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the CutoutFilter with the given configuration.

        :param cfg: Configuration object containing filtering criteria.
        """
        self.cfg = cfg

    def _check_range(self, value: float, min_val: float, max_val: float) -> bool:
        """Check if a value falls within the specified range."""
        return min_val <= value <= max_val

    def _check_exact(self, value: Any, target: Any) -> bool:
        """Check if a value exactly matches the target."""
        return value == target

    def _check_list(self, value: Any, target_list: List[Any]) -> bool:
        """Check if a value is within the target list."""
        return value in target_list

    def filter_cutout(self, cutout_metadata: Dict[str, Any]) -> bool:
        """
        Filter a single cutout based on the configuration criteria.

        :param cutout_metadata: Dictionary containing the cutout metadata.
        :return: True if the cutout matches all filtering criteria, False otherwise.
        """
        # Check morphological properties
        morph_props = cutout_metadata.get("cutout_props", {})
        morph_config = self.cfg.cutout_filters.morphological

        for prop, settings in morph_config.items():
            if isinstance(settings, dict):  # Range-based settings
                if not self._check_range(morph_props.get(prop, 0), settings['min'], settings['max']):
                    return False
            elif isinstance(settings, bool):  # Boolean settings
                if morph_props.get(prop) != settings:
                    return False

        # Check category properties
        category = cutout_metadata.get("category", {})
        category_config = self.cfg.cutout_filters.category

        for prop, target in category_config.items():
            if target and not self._check_exact(category.get(prop), target):
                return False

        return True


class RecipeCreator:
    """Class to create synthetic image recipes based on filtered cutouts."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the RecipeCreator with the given filter configuration.

        :param cfg: Configuration object containing filtering criteria.
        """
        self.filter = CutoutFilter(cfg)
        self.cfg = cfg
        self.recipes = []

    def create_synthetic_image(self, background_image_id: str, cutouts: List[Dict[str, Any]]) -> None:
        """
        Create a synthetic image with the specified background and cutouts.

        :param background_image_id: Background image ID.
        :param cutouts: List of cutouts to include in the synthetic image.
        """
        synthetic_image = {
            "background_image_id": background_image_id,
            "cutouts": cutouts
        }
        self.recipes.append(synthetic_image)
        log.info(f"Synthetic image with background {background_image_id} created with {len(cutouts)} cutouts.")

    def save_recipes(self, output_path: str) -> None:
        """
        Save the created recipes to a JSON file.

        :param output_path: File path where the recipe JSON will be saved.
        """
        with open(output_path, 'w') as f:
            json.dump({"synthetic_images": self.recipes}, f, indent=4)
        log.info(f"Recipes saved to {output_path}")


class CutoutManager:
    """Main class to manage the cutout filtering and recipe creation process."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the CutoutManager with the paths to the configuration, cutouts, and output directory.

        :param cfg: Configuration object containing filtering criteria and paths.
        """
        self.cfg = cfg
        self.cutouts_dir = cfg.paths.cutoutdir    
        self.output_dir = Path(cfg.paths.recipesdir)
        self.background_images_dir = Path(cfg.paths.backgrounddir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recipe_creator = RecipeCreator(self.cfg)
        self.total_images = cfg.cutout_filters.total_images

    def process_cutouts(self) -> None:
        """
        Process each cutout one by one, applying the filtering criteria and adding matching cutouts to the recipe.
        """
        # Load all background images from the specified directory
        background_images = [f.name for f in self.background_images_dir.glob("*") if f.is_file()]

        # Ensure we have enough background images
        if not background_images:
            log.error("No background images found.")
            return

        cutout_files = [f for f in os.listdir(self.cutouts_dir) if f.endswith('.json')]
        random.shuffle(cutout_files)  # Randomize cutout order

        images_created = 0

        while images_created < self.total_images and cutout_files:
            # Randomly select a background image
            background_image_id = random.choice(background_images)

            num_cutouts = random.randint(self.cfg.cutout_filters.cuts_n_image['min'], self.cfg.cutout_filters.cuts_n_image['max'])
            selected_cutouts = []

            for filename in cutout_files[:]:
                if len(selected_cutouts) >= num_cutouts:
                    break

                file_path = os.path.join(self.cutouts_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        metadata = json.load(f)
                        log.info(f"Processing metadata from {filename}")

                        if self.recipe_creator.filter.filter_cutout(metadata):
                            selected_cutouts.append(metadata)
                            cutout_files.remove(filename)  # Remove the used cutout

                except Exception as e:
                    log.error(f"Error processing file {filename}: {e}")

            if selected_cutouts:
                self.recipe_creator.create_synthetic_image(background_image_id, selected_cutouts)
                images_created += 1

        # Save the recipes to the output directory
        recipe_filename = f"recipe_{len(os.listdir(self.output_dir)) + 1}.json"
        output_path = os.path.join(self.output_dir, recipe_filename)
        self.recipe_creator.save_recipes(output_path)


def main(cfg: DictConfig) -> None:
    log.info("Starting synthetic image generation.")
    cutout_manager = CutoutManager(cfg)
    cutout_manager.process_cutouts()
    log.info("Synthetic image generation completed.")
