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


def _latex_sup_sub_to_unicode(latex):
    result = latex
    # Below is subscript and superscript handling
    # Get indices for all `_` and `^`
    subscript_indices = [m.start() for m in re.finditer(r'_', result)]
    superscript_indices = [m.start() for m in re.finditer(r'\^', result)]

    subscript_and_superscript_indices = sorted(subscript_indices + superscript_indices, reverse=True)
    todos = []
    for i in subscript_and_superscript_indices:
        # Are we at the end of the string?
        if i == len(result) - 1:
            # If so, we can't have a subscript or superscript here
            continue

        # Okay, we have a `_` or `^` at index `i`. Now we need to know the
        # extent of the group it applies to

        # Case 1: We have a group like `_{...}` or `^{...}`
        if i + 1 < len(result) and result[i + 1] == '{':
            # We need to be careful, there might be nested braces
            end_brace = i + 2
            brace_count = 1
            while end_brace < len(result) and brace_count > 0:
                if result[end_brace] == '{':
                    brace_count += 1
                elif result[end_brace] == '}':
                    brace_count -= 1
                end_brace += 1
            # Okay, we have the full group from `i + 2` to `end_brace - 1`
            group = result[i + 2:end_brace - 1]
            # We need to remember the start and end indices for the text we need
            # to replace as well as the group of text we need to put in the subscript or superscript
            todos.append((i, end_brace - 1, group))
        else:
            # Case 2: We have a simple `_` or `^` followed by a single character
            if i + 1 < len(result):
                group = result[i + 1]
                todos.append((i, i + 1, group))
    
    # Now we can process the todos in reverse order to avoid messing up indices
    for start, end, group in todos:
        group_unicode = ""
        if result[start] == '_':
            # Replace each character in `group` with its subscript Unicode equivalent
            subscript_map = {
                '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', 
                '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
                'a': 'ₐ', 'e': 'ₑ', 'i': 'ᵢ', 'j': 'ⱼ', 'o': 'ₒ', 
                'r': 'ᵣ', 't': 'ₜ', 'u': 'ᵤ', 'v': 'ᵥ', 'x': 'ₓ',
                '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎'
            }
            
            # Try to convert each character to subscript
            can_convert_all = True
            for char in group:
                if char not in subscript_map:
                    can_convert_all = False
                    break
                group_unicode += subscript_map[char]
            
            # If we can convert all characters to subscripts, use Unicode
            # Otherwise, fall back to HTML tags
            if can_convert_all:
                result = result[:start] + group_unicode + result[end + 1:]
            else:
                result = result[:start] + f"<sub>{group}</sub>" + result[end + 1:]
                
        elif result[start] == '^':
            # Replace each character in `group` with its superscript Unicode equivalent
            superscript_map = {
                '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', 
                '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'd': 'ᵈ', 'e': 'ᵉ', 
                'f': 'ᶠ', 'g': 'ᵍ', 'h': 'ʰ', 'i': 'ⁱ', 'j': 'ʲ', 
                'k': 'ᵏ', 'l': 'ˡ', 'm': 'ᵐ', 'n': 'ⁿ', 'o': 'ᵒ', 
                'p': 'ᵖ', 'r': 'ʳ', 's': 'ˢ', 't': 'ᵗ', 'u': 'ᵘ',
                'v': 'ᵛ', 'w': 'ʷ', 'x': 'ˣ', 'y': 'ʸ', 'z': 'ᶻ',
                '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾'
            }
            
            # Try to convert each character to superscript
            can_convert_all = True
            for char in group:
                if char not in superscript_map:
                    can_convert_all = False
                    break
                group_unicode += superscript_map[char]
            
            # If we can convert all characters to superscripts, use Unicode
            # Otherwise, fall back to HTML tags
            if can_convert_all:
                result = result[:start] + group_unicode + result[end + 1:]
            else:
                result = result[:start] + f"<sup>{group}</sup>" + result[end + 1:]
    
    return result


def latex_to_unicode(latex):
    """Convert inline LaTeX to Unicode characters.
    Takes a string representing inline latex equation and returns
    a string of unicode characters roughly equivalent to the latex.
    
    Args:
        latex (str): The inline LaTeX equation
        
    Returns:
        str: Unicode representation of the equation
    """
    # # Special case for norm expressions with both subscript and superscript
    # if '\\lVert' in latex and '\\rVert' in latex and '_' in latex and '^' in latex:
    #     # For these complex expressions, HTML tags often render better
    #     try:
    #         processed_latex = latex
            
    #         # First replace norm symbols
    #         norm_replacements = {
    #             r'\\lVert': '‖', r'\\rVert': '‖', r'\\Vert': '‖',
    #             r'\\vert': '|', r'\\lvert': '|', r'\\rvert': '|',
    #             r'\\\|': '‖', r'\\norm': '‖',
    #             r'\\Theta': 'Θ', r'\\theta': 'θ'
    #         }
            
    #         for pattern, replacement in norm_replacements.items():
    #             processed_latex = re.sub(pattern, replacement, processed_latex)
            
    #         return processed_latex
    #     except Exception as e:
    #         print(f"Error in special case handling: {e}")
    #         # If that fails, continue with the normal process
    #         pass
        
    # Define mappings from LaTeX to Unicode
    replacements = {
        # Greek letters
        r'\\alpha': 'α', r'\\beta': 'β', r'\\gamma': 'γ', r'\\delta': 'δ',
        r'\\epsilon': 'ε', r'\\varepsilon': 'ε', r'\\zeta': 'ζ', r'\\eta': 'η',
        r'\\theta': 'θ', r'\\vartheta': 'ϑ', r'\\iota': 'ι', r'\\kappa': 'κ',
        r'\\lambda': 'λ', r'\\mu': 'μ', r'\\nu': 'ν', r'\\xi': 'ξ',
        r'\\pi': 'π', r'\\varpi': 'ϖ', r'\\rho': 'ρ', r'\\varrho': 'ϱ',
        r'\\sigma': 'σ', r'\\varsigma': 'ς', r'\\tau': 'τ', r'\\upsilon': 'υ',
        r'\\phi': 'φ', r'\\varphi': 'ϕ', r'\\chi': 'χ', r'\\psi': 'ψ',
        r'\\omega': 'ω',
        r'\\Gamma': 'Γ', r'\\Delta': 'Δ', r'\\Theta': 'Θ', r'\\Lambda': 'Λ',
        r'\\Xi': 'Ξ', r'\\Pi': 'Π', r'\\Sigma': 'Σ', r'\\Upsilon': 'Υ',
        r'\\Phi': 'Φ', r'\\Psi': 'Ψ', r'\\Omega': 'Ω',
        
        # Binary operators
        r'\\pm': '±', r'\\mp': '∓', r'\\times': '×', r'\\div': '÷',
        r'\\cdot': '⋅', r'\\ast': '∗', r'\\star': '⋆', r'\\circ': '∘',
        r'\\bullet': '•', r'\\oplus': '⊕', r'\\ominus': '⊖', r'\\otimes': '⊗',
        r'\\oslash': '⊘', r'\\odot': '⊙', r'\\dagger': '†', r'\\ddagger': '‡',
        r'\\cap': '∩', r'\\cup': '∪', r'\\vee': '∨', r'\\wedge': '∧',
        r'\\setminus': '\\\\', r'\\sqcap': '⊓', r'\\sqcup': '⊔', r'\\uplus': '⊎',
        
        # Relation operators
        r'\\leq': '≤', r'\\prec': '≺', r'\\preceq': '⪯', r'\\ll': '≪',
        r'\\subset': '⊂', r'\\subseteq': '⊆', r'\\sqsubset': '⊏', r'\\sqsubseteq': '⊑',
        r'\\in': '∈', r'\\vdash': '⊢', r'\\smile': '⌣', r'\\geq': '≥',
        r'\\succ': '≻', r'\\succeq': '⪰', r'\\gg': '≫', r'\\supset': '⊃',
        r'\\supseteq': '⊇', r'\\sqsupset': '⊐', r'\\sqsupseteq': '⊒',
        r'\\ni': '∋', r'\\dashv': '⊣', r'\\frown': '⌢', r'\\neq': '≠',
        r'\\sim': '∼', r'\\approx': '≈', r'\\equiv': '≡', r'\\cong': '≅',
        
        # Arrow symbols
        r'\\leftarrow': '←', r'\\rightarrow': '→', r'\\to': '→',
        r'\\leftrightarrow': '↔', r'\\uparrow': '↑', r'\\downarrow': '↓',
        r'\\updownarrow': '↕', r'\\Leftarrow': '⇐', r'\\Rightarrow': '⇒',
        r'\\Leftrightarrow': '⇔', r'\\Uparrow': '⇑', r'\\Downarrow': '⇓',
        r'\\Updownarrow': '⇕', r'\\mapsto': '↦',
        
        # Miscellaneous symbols
        r'\\infty': '∞', r'\\forall': '∀', r'\\exists': '∃', r'\\nexists': '∄',
        r'\\emptyset': '∅', r'\\nabla': '∇', r'\\partial': '∂', r'\\ldots': '…',
        r'\\cdots': '⋯', r'\\vdots': '⋮', r'\\ddots': '⋱', r'\\surd': '√',
        r'\\sqrt': '√',
        
        # Norm symbols
        r'\\lVert': '‖', r'\\rVert': '‖', r'\\Vert': '‖', r'\\vert': '|',
        r'\\lvert': '|', r'\\rvert': '|', r'\\\|': '‖', r'\\norm': '‖',
        
        # Common mathematical symbols
        r'\\sum': '∑', r'\\prod': '∏', r'\\int': '∫', r'\\iint': '∬', 
        r'\\iiint': '∭', r'\\oint': '∮', r'\\therefore': '∴', r'\\because': '∵',
        r'\\propto': '∝', r'\\angle': '∠', r'\\measuredangle': '∡', r'\\sphericalangle': '∢',
        r'\\perp': '⊥', r'\\parallel': '∥', r'\\nparallel': '∦',
        
        # Fractions
        r'\\frac\{1\}\{2\}': '½', r'\\frac\{1\}\{3\}': '⅓', r'\\frac\{2\}\{3\}': '⅔',
        r'\\frac\{1\}\{4\}': '¼', r'\\frac\{3\}\{4\}': '¾', r'\\frac\{1\}\{5\}': '⅕',
        r'\\frac\{2\}\{5\}': '⅖', r'\\frac\{3\}\{5\}': '⅗', r'\\frac\{4\}\{5\}': '⅘',
        r'\\frac\{1\}\{6\}': '⅙', r'\\frac\{5\}\{6\}': '⅚', r'\\frac\{1\}\{8\}': '⅛',
        r'\\frac\{3\}\{8\}': '⅜', r'\\frac\{5\}\{8\}': '⅝', r'\\frac\{7\}\{8\}': '⅞'
    }
    
    # If no special cases matched, continue with general replacements
    result = latex
    for pattern, replacement in replacements.items():
        result = re.sub(pattern, replacement, result)

    # Handle special case for square root to replace curly braces with parentheses
    # Match √{...} patterns
    sqrt_pattern = r'√\{([^{}]+)\}'
    result = re.sub(sqrt_pattern, r'√(\1)', result)
    
    # Also handle nested braces in square roots
    while re.search(r'√\{', result):
        # Find the opening brace position
        start_idx = result.find('√{')
        if start_idx == -1:
            break
            
        # Find matching closing brace
        brace_count = 1
        end_idx = start_idx + 2  # Start after the "√{"
        while end_idx < len(result) and brace_count > 0:
            if result[end_idx] == '{':
                brace_count += 1
            elif result[end_idx] == '}':
                brace_count -= 1
            end_idx += 1
            
        # Replace the braces with parentheses
        if brace_count == 0:
            content = result[start_idx+2:end_idx-1]
            result = result[:start_idx+1] + '(' + content + ')' + result[end_idx:]

    # Handle subscript and superscript
    result = _latex_sup_sub_to_unicode(result)
    
    # Clean up unnecessary spaces that might be introduced by LaTeX command replacements
    # Remove spaces after symbols like ∇, ∂, etc., that should be adjacent to the following character
    # First compile a pattern of all mathematical symbols that should not have spaces after them
    math_symbols = '∇∂∫∬∭∮∑∏√±×÷⋅∗⋆∘•⊕⊖⊗⊘⊙†‡∩∪∨∧⊓⊔⊎≤≥≺≻⪯⪰≪≫⊂⊃⊆⊇⊏⊐⊑⊒∈∋⊢⊣⌣⌢≠∼≈≡≅←→↔↑↓↕⇐⇒⇔⇑⇓⇕↦∞∀∃∄∅‖|'
    result = re.sub(f'([{re.escape(math_symbols)}])\\s+', r'\1', result)
    
    # Also handle common function names that should not have spaces after them
    function_names = ['sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'arcsin', 'arccos', 'arctan', 
                     'sinh', 'cosh', 'tanh', 'log', 'ln', 'exp', 'lim', 'min', 'max']
    for func in function_names:
        # Replace function names followed by space with just the function name
        result = re.sub(f'{func}\\s+', func, result)
    
    # Remove spaces between variables and other variables or between variables and parentheses
    # This mimics LaTeX's behavior where spaces in math mode are generally ignored
    greek_letters = 'αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩ'
    variables = f'[a-zA-Z0-9{greek_letters}]'
    # Remove spaces between variables: "α λ" -> "αλ"
    result = re.sub(f'({variables})\\s+({variables})', r'\1\2', result)
    
    # Remove spaces around operators within expressions: "1 - α λ" -> "1-αλ"
    result = re.sub(r'\\s*([+\-*/=×÷±])\\s*', r'\1', result)
    
    # Remove spaces between variables and parentheses: "Θ (" -> "Θ("
    result = re.sub(f'({variables})\\s+([(])', r'\1\2', result)
    result = re.sub(f'([)])\\s+({variables})', r'\1\2', result)
    
    return result

def extract_inline_latex(content):
    """Extract inline LaTeX from markdown"""
    # Pattern for inline LaTeX: $ ... $ (not preceded or followed by another $)
    # This avoids matching block equations that use $$
    inline_pattern = r'(?<!\$)\$([^\$]+?)\$(?!\$)'
    
    # Find all inline LaTeX
    matches = re.finditer(inline_pattern, content)
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


def find_and_replace_inline_latex(content):
    # Extract inline LaTeX equations from the content with block equations already replaced
    inline_equations = extract_inline_latex(content)
    
    # Process inline equations in reverse order
    for eq in reversed(inline_equations):
        # Convert to Unicode
        unicode_text = latex_to_unicode(eq['latex'])
        
        # Replace inline equation with Unicode
        start = eq['start']
        end = eq['end']
        content = content[:start] + unicode_text + content[end:]
    return content


def latex_to_medium(input_file, output_dir):
    """Convert LaTeX in markdown to Medium format"""
    # Read input file
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Create output directories
    img_dir = setup_dirs(output_dir)
    
    # Extract block LaTeX equations
    block_equations = extract_latex(content)
    
    # Process content
    new_content = content
    
    # Replace block equations in reverse order (from end to beginning)
    # This way we don't need to track offsets
    for i, eq in reversed(list(enumerate(block_equations))):
        # Generate unique filename
        hash_obj = hashlib.md5(eq['latex'].encode())
        filename = hash_obj.hexdigest()

        # Prefix with three digit sequence number to make it easier to put these
        # images in the correct order in the markdown
        seq_num = f"{i:03d}"
        filename = f"{seq_num}_{filename}"

        # Convert to PNG
        png_path = latex_to_png(eq['latex'], img_dir, filename)
        
        if png_path:
            # Use relative path in markdown
            rel_path = os.path.relpath(png_path, output_dir)
            # Create image reference
            img_ref = f"\n\n![Equation]({rel_path})\n\n"
            
            # Replace equation with image reference
            start = eq['start']
            end = eq['end']
            new_content = new_content[:start] + img_ref + new_content[end:]
    
    # Replace inline LaTeX with Unicode
    new_content = find_and_replace_inline_latex(new_content)
    
    # Write output file
    output_file = os.path.join(output_dir, 'medium_ready.md')
    with open(output_file, 'w') as f:
        f.write(new_content)
    
    print(f"Converted file saved to {output_file}")
    print(f"Images saved to {img_dir}")
    print(f"Inline equations converted to Unicode where possible")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert LaTeX in Markdown to Medium format')
    parser.add_argument('input_file', help='Input markdown file')
    parser.add_argument('--output-dir', '-o', default='medium_output', help='Output directory')
    
    args = parser.parse_args()
    latex_to_medium(args.input_file, args.output_dir)
