import cv2
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
import json

def load_class_rgb_mapping(json_file: Path) -> dict:
    """
    Load a JSON file and extract a mapping from class_id to its RGB color.
    
    The JSON file is expected to have a structure where each synthetic image
    contains cutouts and each cutout has a "category" with a "class_id" and an "rgb" list.
    
    Args:
        json_file (Path): Path to the JSON file.
        
    Returns:
        dict: A dictionary mapping class_id (int) to an RGB tuple (R, G, B).
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    
    class_rgb_mapping = {}
    
    # Get the list of synthetic images.
    synthetic_images = data.get("synthetic_images", [])
    for image in synthetic_images:
        cutouts = image.get("cutouts", [])
        for cutout in cutouts:
            category = cutout.get("category", {})
            class_id = category.get("class_id")
            rgb = category.get("rgb")
            
            # Only add the mapping if both class_id and rgb exist.
            if class_id is not None and rgb is not None:
                # Convert the list to a tuple for immutability.
                rgb_tuple = tuple(rgb)
                # If the class_id is already mapped, you might want to check for consistency.
                if class_id in class_rgb_mapping:
                    if class_rgb_mapping[class_id] != rgb_tuple:
                        print(f"Warning: Inconsistent RGB values for class_id {class_id}. "
                              f"Existing: {class_rgb_mapping[class_id]}, New: {rgb_tuple}")
                else:
                    class_rgb_mapping[class_id] = rgb_tuple
    return class_rgb_mapping

def colorize_mask_fixed(mask: np.ndarray, color_map: dict) -> np.ndarray:
    """
    Colorize a mask image using a fixed colormap.
    
    Args:
        mask (np.ndarray): 2D array containing class labels.
        color_map (dict): Fixed mapping from class label to BGR color tuple.
    
    Returns:
        np.ndarray: A 3-channel color image.
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label, color in color_map.items():
        colored[mask == label] = color
        
    return colored

@hydra.main(version_base="1.2", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    project_name = cfg.project_name
    sub_name = cfg.sub_name
    resultsdir = Path(cfg.paths.resultsdir)

    recipe = Path(cfg.paths.recipesdir, f"{project_name}_{sub_name}.json")
    class_rgb_mapping = load_class_rgb_mapping(recipe)
    

    input_mask_dir = resultsdir / "semantic_masks"
    output_mask_dir = resultsdir / "colorized_masks"
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    
    inputs_masks = list(input_mask_dir.glob("*.png"))
    
    for input_mask in inputs_masks:
        output_mask = output_mask_dir / input_mask.name
        
        # Read the grayscale mask. Ensure the image loads in grayscale mode.
        mask = cv2.imread(input_mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load image: {input_mask}")
            return

        colored_mask = colorize_mask_fixed(mask, class_rgb_mapping)
        
        # Save the resulting image.
        cv2.imwrite(output_mask, colored_mask)
        print(f"Colored mask saved to {output_mask}")

if __name__ == "__main__":
    main()
