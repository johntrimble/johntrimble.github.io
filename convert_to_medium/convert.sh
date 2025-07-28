#!/bin/bash
# A simple script to convert LaTeX blog posts to Medium-friendly format

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_FILE="$1"
OUTPUT_DIR="$2"

# Use default values if not provided
if [ -z "$INPUT_FILE" ]; then
  INPUT_FILE="$ROOT_DIR/_drafts/weight-decay-is-not-l-regularization.md"
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$ROOT_DIR/medium_output"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
  echo "Error: Node.js is required but not installed."
  exit 1
fi

# Install required Node.js packages if not already installed
if [ ! -d "$SCRIPT_DIR/node_modules/mathjax-node" ] || [ ! -d "$SCRIPT_DIR/node_modules/sharp" ]; then
  echo "Installing required Node.js packages..."
  cd "$SCRIPT_DIR" && npm install
fi

# Run the Python script
echo "Converting LaTeX equations in $INPUT_FILE to Medium format..."
python3 "$SCRIPT_DIR/latex_to_medium.py" "$INPUT_FILE" --output-dir "$OUTPUT_DIR"

echo "✅ Conversion complete!"
echo "Output files:"
echo "  Markdown: $OUTPUT_DIR/medium_ready.md"
echo "  Images: $OUTPUT_DIR/images/"
