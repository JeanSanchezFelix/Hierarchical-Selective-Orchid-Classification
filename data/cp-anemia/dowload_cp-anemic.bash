#!/bin/bash

# Set download URL and destination directory
URL="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/m53vz6b7fx-1.zip"
DEST_DIR="C:\\Users\\sebastian.cruz6\\Workspace\\Research\\CAWT-URFI\\cp-anemia-detection\\data\\cp-anemia"

# Create destination directory if it doesn't exist
mkdir -p $DEST_DIR

# Download the dataset
echo "Downloading dataset..."
curl -L $URL -o "$DEST_DIR\\cp-anemia.zip"

# Unzip the dataset
echo "Unzipping dataset..."
unzip "$DEST_DIR\\cp-anemia.zip" -d $DEST_DIR

# Remove the zip file
echo "Cleaning up..."
rm "$DEST_DIR\\cp-anemia.zip"

echo "Download and extraction completed."
