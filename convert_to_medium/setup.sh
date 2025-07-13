#!/bin/bash

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Setting up LaTeX to Medium converter..."

# Install Python dependencies
echo "Installing Python dependencies..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-markdown python3-bs4 python3-html2text

# Install latex dependencies
echo "Installing LaTeX dependencies..."
sudo apt-get update && sudo apt-get install -y \
  texlive-latex-base \
  texlive-latex-extra \
  texlive-fonts-recommended \
  dvipng \
  texlive-extra-utils

# Install Python requirements
echo "Installing Python requirements..."
pip3 install -r "$SCRIPT_DIR/requirements.txt"

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
cd "$SCRIPT_DIR" && npm install

echo "Setup complete! You can now run ./convert.sh with your markdown file."
