#!/usr/bin/env python3
"""
A simplified version of the LaTeX to Medium converter.
This script takes a markdown file with LaTeX equations and converts it for Medium:
1. Block LaTeX equations are converted to PNG images
2. Inline LaTeX is converted to Unicode where possible
"""

import re
import os
import sys
import hashlib
import subprocess
import argparse
from pathlib import Path

def setup_dirs(output_dir):
    """Create output directories"""
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    return img_dir

def latex_to_png(latex, output_path, filename):
    """Convert LaTeX to PNG using the new Python script."""
    png_path = os.path.join(output_path, f"{filename}@2x.png")

    # Skip if already exists
    if os.path.exists(png_path):
        return png_path

    # Call the new Python script
    print(f"Converting LaTeX to PNG: {filename}")
    print("Input LaTeX:")
    print(latex)
    try:
        subprocess.run([
            'python3',
            'convert_latex_equation.py',
            '--px-size', '20',
            '-o', png_path
        ], input=latex.encode('utf-8'), check=True, cwd=os.path.dirname(__file__))
    except subprocess.CalledProcessError:
        print(f"Error converting equation to PNG")
        return None

    return png_path

def extract_latex(content):
    """Extract block LaTeX from markdown"""
    # Pattern for block LaTeX: $$ ... $$
    block_pattern = r'\$\$(.*?)\$\$'
    
    # Find all block LaTeX
    matches = re.finditer(block_pattern, content, re.DOTALL)
    equations = []
    
    for match in matches:
        start, end = match.span()
        equations.append({
            'start': start,
            'end': end,
            'latex': match.group(1).strip(),
            'original': match.group(0)
        })
    
    return equations

def simple_latex_to_medium(input_file, output_dir):
    """Convert LaTeX in markdown to Medium format"""
    # Read input file
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Create output directories
    img_dir = setup_dirs(output_dir)
    
    # Extract LaTeX equations
    equations = extract_latex(content)
    
    # Process content
    new_content = content
    offset = 0
    
    # Replace each equation with image reference
    for eq in equations:
        # Generate unique filename
        hash_obj = hashlib.md5(eq['latex'].encode())
        filename = hash_obj.hexdigest()
        
        # Convert to PNG
        png_path = latex_to_png(eq['latex'], img_dir, filename)
        
        if png_path:
            # Use relative path in markdown
            rel_path = os.path.relpath(png_path, output_dir)
            # Create image reference
            img_ref = f"\n\n![Equation]({rel_path})\n\n"
            
            # Replace equation with image reference
            start = eq['start'] + offset
            end = eq['end'] + offset
            new_content = new_content[:start] + img_ref + new_content[end:]
            
            # Update offset
            offset += len(img_ref) - (end - start)
    
    # Write output file
    output_file = os.path.join(output_dir, 'medium_ready.md')
    with open(output_file, 'w') as f:
        f.write(new_content)
    
    print(f"Converted file saved to {output_file}")
    print(f"Images saved to {img_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert LaTeX in Markdown to Medium format')
    parser.add_argument('input_file', help='Input markdown file')
    parser.add_argument('--output-dir', '-o', default='medium_output', help='Output directory')
    
    args = parser.parse_args()
    simple_latex_to_medium(args.input_file, args.output_dir)
