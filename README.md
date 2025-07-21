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
  analysis:
    - analyze_cutouts
    - analyze_preprocessed_cutouts
  move_cutouts: 
  preprocess_cutouts:
  synthesize:

move_cutouts:
  parallel: True
  parallel_workers: 8

preprocess_cutouts:
  remove_soil: 
    Hairy vetch: 20 

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
Handles the analysis of the generated recipe before proceeding with downloading cutouts and generating synthetic images. The purpose of this step is to gain insight into the cutouts and their metadata, allowing informed decisions before committing to the full pipeline. If multiple reports are requested in a single run, they will be combined into one PDF.

#### **2.1 Analyze Cutouts** (`analyze_cutouts.py`)

Produces a report that summarizes the metadata of the cutouts specified in the recipe and compares it against the metadata of all available cutouts for the selected species. This helps assess the representativeness and quality of the selected data. Expected outputs include various graphs and visual summaries of the metadata.

![Bounding Box Graph](markdown_images/bbox_area_metadata_graph.png)

#### Output:
- `projects/<project>/<name>/analysis/report<date>.pdf`

#### **2.2 Analyze Preprocessed Cutouts** (`analyze_preprocessed_cutouts.py`)
Generates a report that explores the range of preprocessing values applicable to the cutouts. Currently, the only supported preprocessing method is EXG. For each species and their associated preprocessing requests, relevant graphs are generated prior to applying preprocessing. This allows evaluation of optimal parameter ranges and helps avoid unnecessary processing on unsuitable data. Note the report generates plots for species based on the ones listed in preprocess_cutouts. 

![Hairy Vetch EXG Plot](markdown_images/hairy_vetch_exg.png)

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

- **Remove_Soil**: Applies the Excess Green Index (EXG) to all cutouts of a specified species, with the goal of minimizing the presence of soil in the images. Note: While EXG is effective at reducing soil visibility, it may also unintentionally remove other plant parts such as stems and flowers. Use with caution. A working range is set on this filter between EXG 0,100

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