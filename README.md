# SemiF-SyntheticPipeline Documentation

## Overview
SemiF-SyntheticPipeline is a Python-based pipeline for generating synthetic images of AgIR data by compositing plant cutouts onto background images. It is designed with configurable filters, image transformations, and metadata management.


## Known Issues and ToDos

### **YOLO Contour Labels Accuracy and Format Update**
- **Issue**: The current implementation of `yolo_contour_labels` is uncertain in terms of accuracy, and it is unclear if the contours are correctly formatted for YOLO segmentation.
- **Current Status**: This setting is **not recommended for use** in its current state.
- **Planned Update**:
  - The `yolo_contour_labels` output should be updated to generate COCO-style polygon annotations instead of YOLO contours.
  - This will improve compatibility with existing COCO-based datasets and annotation tools.

## Installation and Setup

### Prerequisites
- **Python**: Ensure Python (>=3.11) is installed.
- **Conda** (recommended): Used for environment management.

### Install Conda
1. Download Miniconda from [Miniconda website](https://docs.anaconda.com/free/miniconda/).
2. Follow installation instructions for your OS.
3. Verify installation by running:
   ```bash
   conda list
   ```

### Set Up Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/SemiF-SyntheticPipeline.git
   cd SemiF-SyntheticPipeline
   ```
2. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate <env_name>
   ```
3. **Download the Database Locally**
   The pipeline relies on an SQLite database. You need to download it using the provided `copy_db.sh` script run from the repo root:
   ```bash
   bash copy_db.sh
   ```
   Ensure the database is placed in the correct directory as specified in the configuration files.

## Configuration
The pipeline is configured using **Hydra-based YAML files**.

### Main Configuration: `config.yaml`
Defines project details, processing tasks, and key settings:
```yaml
project_name: pm3d
sub_name: test

tasks:
  create_recipes:
  move_cutouts: 
  preprocess_cutouts:
  analysis:
    - analyze_cutouts
    - analyze_preprocessed_cutouts
  synthesize:

move_cutouts:
  parallel: True
  parallel_workers: 8

preprocess_cutouts:
  remove_soil: 
    Hairy vetch: 5  

synthesize:
  resize_factor: 0.35
  parallel: false
  parallel_workers: 4
  instance_masks: False
  yolo_contour_labels: False
  yolo_bbox_labels: True
```

### Cutout Filters: `default.yaml`
Defines filtering criteria for cutouts:
```yaml
morphological:
  non_target_weed: false
  non_target_weed_pred_conf:
    min: 0.9
    max: 1.0

bbox_area_cm2:
  min: 100
  max: 1000
```

## Scripts and Functionality
### **1. Create Recipes** (`create_recipes.py`)
Generates synthetic image recipes by selecting cutouts and assigning them to background images.

#### Features:
- Queries cutout metadata from SQLite.
- Use `conf/cutout_filters/default.yaml` for creating synthetic image recipes.
- Outputs recipes as JSON files.

#### Output:
- `recipes/{project_name}_{sub_name}.json`
  ```json
  {
    "synthetic_images": [
      {
        "synthetic_image_id": "unique_id",
        "background_image_id": "bg_001.jpg",
        "cutouts": [
          { "cutout_id": "cutout_001", "batch_id": "batch_1" }
        ]
      }
    ]
  }
  ```

### **2. Analysis** (`analysis.py`)
Handles the analsis of the generated recipe before commiting to pulling cutouts and generating synthetic images. 

#### **2.1 Analyze Cutouts** (`analyze_cutouts.py`)
Generates a report on the metadata of the cutouts specified in your recipe and compares it to the metadata of all cutouts for the given species.

#### Output:
- `projects/<project>/<name>/analysis/report<date>.pdf`

#### **2.2 Analyze Preprocessed Cutouts** (`analyze_preprocessed_cutouts.py`)
Generates a report on the preprocessing done to the cutouts. 

#### Output:
- `projects/<project>/<name>/analysis/report<date>.pdf`

### **3. Move Cutouts** (`move_cutouts.py`)
Moves cutout images from long-term storage to a local directory.

#### Includes:
- **Parallel download support**
- **Looks into both primary and secondary storage locations**
- **Ensures unique cutouts** before downloading to save time.

#### Output:
- `data/cutouts/*.png` (Downloaded cutout images)

### **4. Preprocess Cutouts** (`preprocess_cutouts.py`)
Preprocesses downloaded cutouts based on the what you set for a certain species.

#### Arguments
- **Remove_Soil. Applies EXG to all cutouts of a certain species, intensity based on level set**

#### Output:
- `data/cutouts/*.png` (Preprcessed cutout images)

### **5. Synthesize** (`synthesize.py`)
Generates synthetic images by overlaying cutouts onto backgrounds.

#### Includes:
- **Parallel processing** with `ProcessPoolExecutor`.
- **Random transformations**: Rotation, flipping, etc.
- **Shadow simulation**: Adjusts shadows based on cutout sizes.
- **Output flexibility**: Saves images, masks, and YOLO labels.

#### Output:
- `projects/<project>/<name>/results/images/*.jpg` (Synthetic images)
- `projects/<project>/<name>/results/semantic_masks/*.png` (Class-based masks)
- `projects/<project>/<name>/results/instance_masks/*.png` (Instance masks, optional)
- `projects/<project>/<name>/results/yolo_bbox_labels/*.txt` (YOLO format labels)

## Running the Pipeline
To execute all tasks:
```bash
python main.py
```

## License
This repository is open-source. You are free to use and modify it. Attribution is appreciated if shared publicly.