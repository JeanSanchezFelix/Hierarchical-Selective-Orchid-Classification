#!/bin/bash
# Script: encode_model.sh
# Purpose: Convert a specified TensorFlow Lite model to an Arduino header file

# Check if the user provided an input argument.
if [ "$#" -ne 1 ]; then
    echo "Error: No tflite model provided"
    echo "Usage: $0 <tflite-model-file>"
    exit 1
fi

# Assign the first command-line argument to MODEL_FILE.
MODEL_FILE="$1"

# Check if the file exists.
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: File '$MODEL_FILE' not found!"
    exit 1
fi

# Optionally, generate the header file name automatically.
# This example uses the model file name but with a .h extension.
HEADER_FILE="${MODEL_FILE%.*}.h"

# Inform the user which files are being processed.
echo "Converting model file '$MODEL_FILE' to header file '$HEADER_FILE'..."

# Start the header file with the array declaration.
echo "const unsigned char model[] = {" > "$HEADER_FILE"

# Convert the TFLite model file to a C-style array and append it to the header file.
xxd -i "$MODEL_FILE" >> "$HEADER_FILE"

# Append the closing brace and semicolon to finish the array declaration.
echo "};" >> "$HEADER_FILE"

# Display the file size for verification.
MODEL_H_SIZE=$(stat -c %s "$HEADER_FILE")
echo "Header file, $HEADER_FILE, is $(printf "%'.0f" "$MODEL_H_SIZE") bytes."
