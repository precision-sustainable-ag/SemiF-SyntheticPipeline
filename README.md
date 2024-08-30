
# SemiF-SyntheticPipeline


## Installation and Setup

### Installing Conda
To manage the project's dependencies efficiently, we use Conda, a powerful package manager and environment manager. Follow these steps to install Conda if you haven't already:

1. Download the appropriate version of Miniconda for your operating system from the official [Miniconda website](https://docs.anaconda.com/free/miniconda/).
2. Follow the installation instructions provided on the website for your OS. This typically involves running the installer from the command line and following the on-screen prompts.
3. Once installed, open a new terminal window and type `conda list` to ensure Conda was installed correctly. You should see a list of installed packages.


### Setting Up Your Environment Using an Environment File
After installing Conda, you can set up an environment for this project using an environment file, which specifies all necessary dependencies. Here's how:

1. Clone this repository to your local machine.
2. Navigate to the repository directory in your terminal.
3. Locate the `environment.yaml` file in the repository. This file contains the list of packages needed for the project.
4. Create a new Conda environment by running the following command:
   ```bash
   conda env create -f environment.yaml
   ```
   This command reads the `environment.yaml` file and creates an environment with the name and dependencies specified within it.

5. Once the environment is created, activate it with:
   ```bash
   conda activate <env_name>
   ```
   Replace `<env_name>` with the name of the environment specified in the `environment.yaml` file.

## Transfer data
The **transfer data script** manages downloading and reporting of cutouts from NCSU's locker storage to a local directory. 


- **Batch Downloading:** Automatically downloads image cutouts from a specified NFS directory to a local directory, with checks to skip batches that have already been fully downloaded.
- **Concurrency:** Utilizes multithreading with a configurable number of workers to speed up the download process.
- **Comprehensive Reporting:** Generates and updates a CSV report that tracks the number of unique cutouts per batch and folder size in MiB of batches that have been downloaded.
- **Logging:** Includes detailed logging to track the progress and status of the downloading and reporting processes.


### Configuration

The script is configured using a YAML or a similar configuration file, typically managed through `omegaconf`. This configuration file should include paths for the NFS storage, local download directory, batch list file, and the report file.


### Usage

1. **Prepare the Configuration:**
   - Ensure the configuration file (`config.yaml`) is set up with the correct paths.

2. **Prepare the Batch List:**
   - The batch list should be a text file (e.g., `batch_list.txt`) containing the names of the batches to be downloaded, with each batch name on a new line.

## Synthetic Image Generator

This script is generates "recipe" JSON files that describe how to create synthetic images by combining cutouts (small image segments) with background images. The cutouts are selected based on specific morphological and categorical criteria defined in a configuration file. The script allows you to control the total number of synthetic image recipes created, the number of cutouts per image, and other filtering criteria to ensure the recipes meet your specific requirements.

- **Cutout Configuration:** Use a YAML configuration file to specify filtering criteria for the cutouts, including morphological properties (e.g., area, eccentricity) and categorical attributes (e.g., genus, species).
- **Criteria:** Define the total number of synthetic images to be created and the range for the number of cutouts per image.
- **Random Cutout Selection:** The script randomly selects cutouts for each synthetic image, ensuring variety and randomness in the generated images.
- **Cutout Reuse:** Each cutout is used only once across all synthetic images, preventing duplication (this may be changed in the future).

### Usage

1. **Prepare Your Configuration File:**
   - Create a cusomt filter configuration file that specifies the paths to your cutout metadata files, background images, and output directory. Define the filtering criteria and the total number of synthetic images you want to generate.

   Default configuration can be found (`conf/cutout_filters/default.yaml`)

3. **Output:**
   - The script will generate recipe JSON files based on the criteria specified in the configuration file. These recipes describe how to create synthetic images and will be saved in the directory specified by `cfg.paths.recipesdir`.

### Notes

- The script makes sure that each cutout is used only once across all synthetic images.
- Background images are selected randomly from the specified directory, and the script will reuse background images if the number of synthetic images requested exceeds the number of available background images.

### Troubleshooting

- **No Background Images Found:** Ensure that the `background_images_dir` path in the configuration file points to a directory containing valid image files.
- **Insufficient Cutouts:** If you have a very small number of cutouts or strict filtering criteria, the script may fail to generate the desired number of synthetic images.

###TODOs:
- add functionality to use cutout with replacement in the event that many high density synthetic images are created and cutouts need to be reused

## License

This script is provided as-is, with no warranties or guarantees. You are free to modify and distribute it as needed. However, attribution is appreciated if you share it publicly.