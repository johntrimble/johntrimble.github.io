#!/usr/bin/env python3
import subprocess
import tempfile
import os
import sys
import argparse
import textwrap


def wrap_snippet(snippet: str) -> str:
    # strip common indent but preserve inner structure
    s = textwrap.dedent(snippet).strip()

    # Do we already use an environment?
    if s.startswith(r"\begin{"):
        return s

    #  Wrap in an equation environment
    s = "\n".join([r"\begin{equation*}", s, r"\end{equation*}"])

    return s


def latex_to_png(latex: str,
                 fontsize: int = 12,
                 output: str = "equation.png",
                 dpi: int = 1200):
    
    # Make sure we have an absolute path for the output
    output = os.path.abspath(output)

    # Double DPI for sharper images
    dpi = int(dpi * 2)

    # 1) Clean up the LaTeX input
    cleaned = wrap_snippet(latex)

    # 2) Assemble a minimal standalone .tex with no stray indentation
    tex_lines = [
        f"\\documentclass[varwidth,border=1pt,{fontsize}pt]{{standalone}}",
        "\\usepackage{amsmath,amssymb,amsfonts}",
        "\\pagestyle{empty}",
        "\\begin{document}",
        cleaned,
        "\\end{document}",
    ]
    tex = "\n".join(tex_lines)

    print("----- LaTeX input -----")
    print(tex)
    print("------------------------")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "eq.tex")
        with open(path, "w") as f:
            f.write(tex)

        # 3) latex → .dvi, halting on the first error
        proc = subprocess.run(
            ["latex", "-halt-on-error", "-interaction=nonstopmode", "eq.tex"],
            cwd=tmp, capture_output=True, text=True
        )
        if proc.returncode != 0:
            print(f"⛔ LaTeX failed (exit code {proc.returncode})", file=sys.stderr)
            print("----- STDOUT -----", file=sys.stderr)
            print(proc.stdout, file=sys.stderr)
            print("----- STDERR -----", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            sys.exit(1)

        # 4) dvipng → .png
        proc2 = subprocess.run(
            [
                "dvipng",
                "-T", "tight",
                "-D", str(dpi),
                "-z", "1",           # 1× magnification → 1pt = DPI/72 px
                "--truecolor",
                "-bg", "Transparent",
                "-o", output,
                "eq.dvi"
            ],
            cwd=tmp, capture_output=True, text=True
        )
        if proc2.returncode != 0:
            print(f"⛔ dvipng failed (exit code {proc2.returncode})", file=sys.stderr)
            print(proc2.stdout, file=sys.stderr)
            print(proc2.stderr, file=sys.stderr)
            sys.exit(1)

        print(f"✔ PNG written to {output}")

def main():
    p = argparse.ArgumentParser(
        description="Render multi-line LaTeX to a 12 pt-scaled PNG."
    )
    p.add_argument("infile", nargs="?", help=".tex snippet file (or stdin if omitted)")
    p.add_argument("-o", "--output", default="equation.png", help="Output PNG name")
    p.add_argument(
        "--px-size", type=int, default=20,
        help="Desired font height in *pixels*.  Renders at 72 dpi so 1pt→1px."
    )
    args = p.parse_args()

    if args.infile:
        latex = open(args.infile).read()
    else:
        latex = sys.stdin.read()

    fontsize_pt = 12
    dpi = int(args.px_size / fontsize_pt * 72)

    latex_to_png(latex, fontsize=fontsize_pt, output=args.output, dpi=dpi)

if __name__ == "__main__":
    main()
