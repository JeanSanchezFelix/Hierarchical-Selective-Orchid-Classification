#!/bin/bash
source ~/.bash_profile

# Script to download monkeypox dataset from https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset

# Define the dataset name
DATASET="nafin59/monkeypox-skin-lesion-dataset"
# Define compressed file name
COMPRESSED_FILE="monkeypox-skin-lesion-dataset.zip"
# Define destinaiton directory
DEST_DIR="download"

# Checks if daaset has been donwloaded
if [ ! -f $COMPRESSED_FILE ]; then

	# Check if kaggle is installed, if not, install it
	if ! command -v kaggle &> /dev/null
	then
    		echo "Kaggle CLI not found, installing..."
    		pip install kaggle
	fi

	# Check if kaggle.json exists in the current directory
	#if [ ! -f "~/.kaggle/kaggle.json" ]; then
	#    echo "kaggle.json not found! Please place your kaggle.json file in the current directory."
	#    exit 1
	#fi


	# Create a .kaggle directory in the home directory if it doesn't exist
	KAGGLE_DIR="$HOME/.kaggle"
	if [ ! -d "$KAGGLE_DIR" ]; then
    		echo "Creating .kaggle directory in home directory..."
    		mkdir -p "$KAGGLE_DIR"
	fi

	# Move kaggle.json to the .kaggle directory
	echo "Moving kaggle.json to $KAGGLE_DIR..."
	mv ../kaggle.json "$KAGGLE_DIR/"

	# Set permissions for kaggle.json
	echo "Setting permissions for kaggle.json..."
	chmod 600 "$KAGGLE_DIR/kaggle.json"


	# Download the dataset
	echo "Downloading dataset $DATASET..."
	kaggle datasets download $DATASET
else
	echo "Zip file found, skipping download..."
fi

# Unzip the downloaded dataset
echo "Unzipping the dataset..."
unzip "$COMPRESSED_FILE" -d "${DEST_DIR}"

mv -v "${DEST_DIR}/Original Images/Original Images" "${DEST_DIR}/Test"
mv -v "${DEST_DIR}/Augmented Images/Augmented Images" "${DEST_DIR}/Train"
mv -v "${DEST_DIR}/Train/Monkeypox_Augmented" "${DEST_DIR}/Train/Monkeypox"
mv -v "${DEST_DIR}/Train/Others_Augmented" "${DEST_DIR}/Train/Others"
mv -v "${DEST_DIR}/Test/Monkey Pox" "${DEST_DIR}/Test/Monkeypox"

echo "Download and extraction complete."
