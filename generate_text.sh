#!/bin/bash

# Usage: ./list_files.sh /path/to/folder /path/to/output.txt prefix

# Folder to scan
FOLDER="$1"

# Output file
OUTPUT="$2"

# Prefix (default: noise_testset_wav)
PREFIX="${3:-noise_testset_wav}"

# Check if folder exists
if [ ! -d "$FOLDER" ]; then
  echo "Error: Folder not found: $FOLDER"
  exit 1
fi

# List files (not directories), one per line, and save to output
find "$FOLDER" -type f -exec basename {} \; | sed "s|^|$PREFIX/|" > "$OUTPUT"

echo "File list saved to $OUTPUT with prefix '$PREFIX/'"
