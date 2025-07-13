# LaTeX to Medium Converter

A toolkit for converting LaTeX-heavy Markdown blog posts for use on Medium, which doesn't support LaTeX natively.

## Features

- Converts block LaTeX equations to PNG images with appropriate sizing
- Replaces inline LaTeX with Unicode where possible
- Outputs a Medium-friendly Markdown file with image references
- Generates images with proper dimensions for readability

## Installation

### Prerequisites

- Python 3.6+
- Node.js 14+
- npm

### Setup

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Install Node.js dependencies:

   ```bash
   npm install
   ```

   Or use the setup script:

   ```bash
   ./setup.sh
   ```

## Usage

### Convert a Post

```bash
./convert.sh ../path/to/your-post.md
```

### Output

The conversion creates two main outputs:
- `../medium_output/medium_ready.md`: The converted Markdown file ready for Medium
- `../medium_output/images/`: Directory containing generated PNG images for LaTeX equations

### Checking Image Sizes

If you need to verify the dimensions of generated PNG files:

```bash
./check_images.sh ../medium_output/images/
```

## How It Works

The conversion process:
1. Extracts block LaTeX equations from your Markdown
2. Converts them to SVG using MathJax
3. Converts SVGs to PNG images with appropriate sizing
4. Replaces the LaTeX with image references in the output Markdown
5. Converts inline LaTeX to Unicode where possible

## Scripts

- `latex_to_medium.py`: Python script for LaTeX conversion
- `convert.sh`: Shell script to run the conversion process
- `setup.sh`: Script to install dependencies
- `check_images.sh`: Utility to check PNG dimensions

## Troubleshooting

If you encounter "module not found" errors, make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
npm install
```
