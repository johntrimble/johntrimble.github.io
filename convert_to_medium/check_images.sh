#!/bin/bash

# This script checks the dimensions of PNG files in the medium_output/images directory
# and reports on their sizes to help verify they are correctly sized for Medium

# Set script directory and default image directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_DIR="${1:-$ROOT_DIR/medium_output/images}"

echo "PNG Image Size Report:"
echo "======================"
echo "Filename                                    | Width x Height"
echo "--------------------------------------------|---------------"

for png_file in "$IMAGE_DIR"/*.png; do
  if [ -f "$png_file" ]; then
    filename=$(basename "$png_file")
    # Get dimensions using file command
    dimensions=$(file "$png_file" | grep -o "[0-9]* x [0-9]*")
    printf "%-44s | %s\n" "$filename" "$dimensions"
  fi
done

echo "======================"
echo "Note: Images should have appropriate dimensions based on equation complexity"
echo "Single line equations: ~600-700px width, ~70-90px height"
echo "Multi-line equations: wider/taller based on content"
