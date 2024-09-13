
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
   conda env create -f environment.yml
   ```
   This command reads the `environment.yaml` file and creates an environment with the name and dependencies specified within it.

5. Once the environment is created, activate it with:
   ```bash
   conda activate <env_name>
   ```
   Replace `<env_name>` with the name of the environment specified in the `environment.yaml` file.

### Setting Up MongoDB and Mongosh

#### Step 1: Download and Install MongoDB
1. **SSH into your server**.
2. **Create a directory** for MongoDB:
   ```bash
   mkdir -p ~/mongodb && cd ~/mongodb
   ```
3. **Download MongoDB binaries** from the [MongoDB Download Center](https://www.mongodb.com/try/download/community) or use `wget`:
   ```bash
   wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-<version>.tgz
   ```
4. **Extract the binaries**:
   ```bash
   tar -zxvf mongodb-linux-x86_64-<version>.tgz
   mv mongodb-linux-x86_64-<version> mongo
   ```

#### Step 2: Configure MongoDB
1. **Add MongoDB to your PATH**:
   ```bash
   echo 'export PATH=~/mongodb/mongo/bin:$PATH' >> ~/.bashrc
   source ~/.bashrc
   ```
2. **Create directories** for MongoDB data and logs:
   ```bash
   mkdir -p ~/mongodb/data ~/mongodb/logs
   ```

#### Step 3: Install Mongosh
1. **Download Mongosh** from the [MongoDB Shell download page](https://www.mongodb.com/try/download/shell) or use `wget`:
   ```bash
   wget https://downloads.mongodb.com/compass/mongosh-<version>-linux-x64.tgz
   ```
2. **Extract and install Mongosh**:
   ```bash
   mkdir ~/bin
   tar -xzvf mongosh-<version>-linux-x64.tgz -C ~/bin/
   ```
3. **Update PATH for Mongosh**:
   ```bash
   echo 'export PATH="$HOME/bin/mongosh-<version>-linux-x64/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

#### Step 4: Running MongoDB
1. **Start MongoDB**:
   ```bash
   mongod --dbpath ~/mongodb/data/db --bind_ip_all --logpath ~/mongodb/logs/mongod.log --fork
   ```
2. **Verify MongoDB is running**:
   ```bash
   ps -aux | grep mongod
   ```

Parts that need access to the mongodb can now access it.

## Scripts:


### Json to Mongo

This script loads JSON data from batch directories in an NFS storage system into a MongoDB database. It reads the batch names from a YAML configuration file, checks both primary and secondary NFS storage locations for the corresponding JSON metadata files, and inserts the data into a specified MongoDB collection.

#### Key Features
- **MongoDB Integration**: Connects to MongoDB to insert JSON data.
- **Batch Processing**: Reads batch names from a YAML configuration and processes the corresponding directories in the NFS storage locker.
- **Primary and Secondary Storage**: Automatically checks both primary and secondary NFS storage paths for the presence of batch directories.

#### Output
- **Data Insertion**: Inserts JSON data from batch directories into the specified MongoDB collection. 

### **Create Recipes**

This script is responsible for creating synthetic image recipes by selecting cutout images based on specific criteria and associating them with background images. The recipes are then saved in JSON format for use in synthetic dataset generation.

#### Key Features
- **MongoDB Integration**: Retrieves cutout metadata from a MongoDB collection based on specific filter criteria defined in the configuration.
- **Randomized Synthetic Image Generation**: Associates cutouts with randomly selected background images and creates synthetic images with varying numbers of cutouts.
- **Flexible Cutout Usage**: Configurable to either reuse cutouts across multiple synthetic images or ensure each cutout is used only once.
- **JSON Output**: Saves the generated synthetic image recipes to a JSON file for further processing.

#### Output
- **Synthetic Image Recipes**: A JSON file containing a list of synthetic images, each with a unique ID, background image, and associated cutouts. The file is saved in the `recipes` directory under the project directory.


### **Move Cutouts**

This script is responsible for downloading plant cutout images from long-term storage to a local directory. It can handle both sequential and concurrent data transfer. The downloaded cutouts are stored locally for further use in synthetic image generation.

#### Key Features
- **Sequential and Parallel Processing**: The script can download cutouts in a sequential manner or use multithreading.
- **Dual Storage Locations**: Looks in both primary and secondary long-term storage locations.

#### Output
- **Downloaded Images**: The script downloads `.png` cutout images to the specified local directory.


### **Synthesize**

This script is designed to generate synthetic images by overlaying plant cutout images onto various backgrounds using a copy-and-paste method. The script provides CPU parallelism.

- **Parallelism**: Utilizes Python's `concurrent.futures.ProcessPoolExecutor` to enable concurrent processing of multiple image recipes, leveraging multi-core CPUs.
- **Transformations**: Applies a variety of image transformations (e.g., rotation, flipping) using the Albumentations library.
- **Dynamic Shadow Generation**: Simulates dynamic shadows for the cutouts based on their size and position relative to the light source.
- **Cutout Distribution**: Supports random placement of cutouts on background images, creating diverse compositions.
- **Output Flexibility**: Saves images, semantic masks, instance masks, and YOLO format segmentation labels.

#### Output
- **Images**: Generated synthetic images in `.jpg` format.
- **Semantic Masks**: Corresponding masks with class annotations in `.png` format.
- **Instance Masks**: Optional masks for instance annotations.
- **YOLO Labels**: Segmentation contours in YOLO format.



## License

This script is provided as-is, with no warranties or guarantees. You are free to modify and distribute it as needed. However, attribution is appreciated if you share it publicly.
