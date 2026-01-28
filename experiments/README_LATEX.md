# LaTeX Generation Script

The `generate_latex.py` script automatically generates LaTeX tables and TikZ plots from experiment results.

## Usage

```bash
uv run python experiments/generate_latex.py
```

## Output Files

The script generates the following files in the `outputs/` directory:

- `experiment1_table.tex` - LaTeX table for Experiment 1 results
- `experiment1_plot.tex` - TikZ plot for Experiment 1 results
- `experiment2_table.tex` - LaTeX table for Experiment 2 results
- `experiment2_plot.tex` - TikZ plot for Experiment 2 results
- `results.tex` - Complete LaTeX document with all tables and plots

## Compiling LaTeX

To compile the full document:

```bash
cd outputs
pdflatex results.tex
```

Or include the individual `.tex` files in your own LaTeX documents.

## Required LaTeX Packages

The generated LaTeX requires the following packages:

- `booktabs` - For professional-looking tables
- `pgfplots` - For TikZ plots
- `geometry` - For page layout (optional, can be removed if not needed)

Add these to your LaTeX preamble:

```latex
\usepackage{booktabs}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
```

## Features

- **Automatic result loading**: Finds the most recent results for each experiment
- **Professional tables**: Uses `booktabs` for clean, publication-ready tables
- **TikZ plots**: Generates publication-quality plots with error bars
- **Complete document**: Generates a standalone LaTeX document ready to compile
