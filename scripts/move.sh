#!/bin/bash

# Define source and destination directories
SOURCE_DIR="$1"
DEST_DIR="/mnt/research-projects/s/screberg/longterm_images2/semif-synthetic-datasets"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Move the pm3d directory itself to the destination using rsync
rsync -av --remove-source-files "$SOURCE_DIR" "$DEST_DIR/"

# Change permissions to make the files usable by all
chmod -R a+rw "$DEST_DIR"

echo "Move and permission change completed successfully."
