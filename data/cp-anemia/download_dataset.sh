#!/bin/bash

source ~/.bash_profile
# Set download URL and destination directory
URL="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/m53vz6b7fx-1.zip"
FILENAME="cp-anemia.zip"
DEST_DIR="download"

unrar_present=`conda list | grep unrar`
if [ -z "$unrar_present" ]; then
	echo "Unrar package not installed, installing it"
	conda install -y unrar
fi

# Download the dataset
if [ ! -f $FILENAME ]; then
	echo "Downloading dataset..."
	curl -L $URL -o $FILENAME
fi
# Unzip the dataset
echo "Unzipping dataset..."
unzip $FILENAME -d "."
# Unrar don't like spaces in the filename, removing them
mv -v "CP-AnemiC (A Conjunctival Pallor) Dataset from Ghana" cp-anemia-dir
mv -v "cp-anemia-dir/CP-AnemiC dataset.rar" "cp-anemia-dir/dataset.rar"
pwd
unrar x "cp-anemia-dir/dataset.rar"

mkdir -p $DEST_DIR
mv Anemic $DEST_DIR
mv Non-anemic $DEST_DIR


# Remove the zip file
echo "Cleaning up..."
rm -rf "cp-anemia-dir" 

echo "Download and extraction completed."
