#!/bin/bash

# Copy the database from longterm_images2 to the data/db directory
SOURCE_DB="/mnt/research-projects/s/screberg/longterm_images2/semifield-database/agir.db"
DEST_DIR="data/db"
DEST_DB="$DEST_DIR/agir.db"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Check if source database exists
if [ ! -f "$SOURCE_DB" ]; then
    echo "Source database not found: $SOURCE_DB"
    exit 1
fi

# Copy the database only if the destination database doesn't exist
if [ -f "$DEST_DB" ]; then
    echo "Destination database already exists: $DEST_DB"
    exit 1
fi

# Copy the database
echo "Copying database from $SOURCE_DB to $DEST_DB"
cp "$SOURCE_DB" "$DEST_DB"

# Verify the copy was successful
if [ $? -eq 0 ]; then
    echo "Database copied successfully to $DEST_DB"
else
    echo "Failed to copy database"
    exit 1
fi