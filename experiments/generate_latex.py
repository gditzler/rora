"""
Generate LaTeX tables and TikZ plots from experiment results.

This script reads pickle files from the outputs/ directory and generates
LaTeX code for tables and TikZ plots showing the performance results.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob


def load_latest_results(experiment_name: str) -> Optional[Dict]:
    """Load the most recent results file for a given experiment."""
    output_dir = Path(__file__).parent.parent / "outputs"
    pattern = f"{experiment_name}_results_*.pkl"
    files = sorted(output_dir.glob(pattern), reverse=True)
    
    if not files:
        print(f"Warning: No results found for {experiment_name}")
        return None
    
    latest_file = files[0]
    print(f"Loading results from: {latest_file.name}")
    
    with open(latest_file, "rb") as f:
        return pickle.load(f)


def generate_experiment1_table(results: Dict) -> str:
    """Generate LaTeX table for Experiment 1 results."""
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{MNIST Multi-Task Learning Results}")
    latex.append("\\label{tab:experiment1}")
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\toprule")
    latex.append("Method & Task A (Even/Odd) & Task B (Bit-Parity) \\\\")
    latex.append("\\midrule")
    
    # Base model
    base_acc = results["results"]["base"]
    latex.append(f"Base (frozen) & {base_acc:.2f}\\% & N/A \\\\")
    
    # RoRA results
    for rank in [4, 8, 16]:
        key = f"rora_rank_{rank}"
        if key in results["results"]:
            r = results["results"][key]
            mean_a, std_a = r["task_a"]
            mean_b, std_b = r["task_b"]
            latex.append(
                f"RoRA Rank {rank} & ${mean_a:.2f} \\pm {std_a:.2f}\\%$ & "
                f"${mean_b:.2f} \\pm {std_b:.2f}\\%$ \\\\"
            )
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_experiment1_plot(results: Dict) -> str:
    """Generate TikZ plot for Experiment 1 results."""
    latex = []
    latex.append("\\begin{tikzpicture}")
    latex.append("\\begin{axis}[")
    latex.append("    width=\\textwidth,")
    latex.append("    height=0.5\\textwidth,")
    latex.append("    xlabel={RoRA Rank},")
    latex.append("    ylabel={Accuracy (\\%)},")
    latex.append("    legend pos=south east,")
    latex.append("    grid=major,")
    latex.append("    xmin=2, xmax=18,")
    latex.append("    ymin=95, ymax=100,")
    latex.append("    xtick={4, 8, 16},")
    latex.append("]")
    
    # Extract data
    ranks = []
    task_a_means = []
    task_a_stds = []
    task_b_means = []
    task_b_stds = []
    
    for rank in [4, 8, 16]:
        key = f"rora_rank_{rank}"
        if key in results["results"]:
            ranks.append(rank)
            r = results["results"][key]
            task_a_means.append(r["task_a"][0])
            task_a_stds.append(r["task_a"][1])
            task_b_means.append(r["task_b"][0])
            task_b_stds.append(r["task_b"][1])
    
    # Task A plot
    coords_a = ", ".join([f"({r},{m:.2f})" for r, m in zip(ranks, task_a_means)])
    latex.append(f"\\addplot[mark=*, blue, thick] coordinates {{{coords_a}}};")
    latex.append("\\addlegendentry{Task A (Even/Odd)}")
    
    # Task B plot
    coords_b = ", ".join([f"({r},{m:.2f})" for r, m in zip(ranks, task_b_means)])
    latex.append(f"\\addplot[mark=square*, red, thick] coordinates {{{coords_b}}};")
    latex.append("\\addlegendentry{Task B (Bit-Parity)}")
    
    # Base model line
    base_acc = results["results"]["base"]
    latex.append(f"\\addplot[dashed, gray, thick] coordinates {{(2,{base_acc:.2f}) (18,{base_acc:.2f})}};")
    latex.append("\\addlegendentry{Base (frozen)}")
    
    latex.append("\\end{axis}")
    latex.append("\\end{tikzpicture}")
    
    return "\n".join(latex)


def generate_experiment2_table(results: Dict) -> str:
    """Generate LaTeX table for Experiment 2 results."""
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{MNIST Novel Class Adaptation (Add-9) Results}")
    latex.append("\\label{tab:experiment2}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("Method & Rank 4 & Rank 8 & Rank 16 \\\\")
    latex.append("\\midrule")
    
    # Base model
    base_acc = results["results"]["base_0_8"]
    latex.append(f"Base (0-8 only) & {base_acc:.2f}\\% & N/A & N/A \\\\")
    latex.append("\\midrule")
    
    # RoRA and LoRA results
    for adapter_type in ["rora", "lora"]:
        adapter_name = adapter_type.upper()
        if adapter_type in results["results"]:
            adapter_results = results["results"][adapter_type]
            r4 = adapter_results.get("rank_4", (0, 0))
            r8 = adapter_results.get("rank_8", (0, 0))
            r16 = adapter_results.get("rank_16", (0, 0))
            latex.append(
                f"{adapter_name} & ${r4[0]:.2f} \\pm {r4[1]:.2f}\\%$ & "
                f"${r8[0]:.2f} \\pm {r8[1]:.2f}\\%$ & "
                f"${r16[0]:.2f} \\pm {r16[1]:.2f}\\%$ \\\\"
            )
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_experiment2_plot(results: Dict) -> str:
    """Generate TikZ plot for Experiment 2 results."""
    latex = []
    latex.append("\\begin{tikzpicture}")
    latex.append("\\begin{axis}[")
    latex.append("    width=\\textwidth,")
    latex.append("    height=0.5\\textwidth,")
    latex.append("    xlabel={RoRA/LoRA Rank},")
    latex.append("    ylabel={Accuracy (\\%)},")
    latex.append("    legend pos=south east,")
    latex.append("    grid=major,")
    latex.append("    xmin=2, xmax=18,")
    latex.append("    ymin=85, ymax=100,")
    latex.append("    xtick={4, 8, 16},")
    latex.append("]")
    
    # Extract data
    ranks = [4, 8, 16]
    
    # RoRA plot
    if "rora" in results["results"]:
        rora_results = results["results"]["rora"]
        rora_means = [rora_results.get(f"rank_{r}", (0, 0))[0] for r in ranks]
        coords_rora = ", ".join([f"({r},{m:.2f})" for r, m in zip(ranks, rora_means)])
        latex.append(f"\\addplot[mark=*, blue, thick] coordinates {{{coords_rora}}};")
        latex.append("\\addlegendentry{RoRA}")
    
    # LoRA plot
    if "lora" in results["results"]:
        lora_results = results["results"]["lora"]
        lora_means = [lora_results.get(f"rank_{r}", (0, 0))[0] for r in ranks]
        coords_lora = ", ".join([f"({r},{m:.2f})" for r, m in zip(ranks, lora_means)])
        latex.append(f"\\addplot[mark=square*, red, thick] coordinates {{{coords_lora}}};")
        latex.append("\\addlegendentry{LoRA}")
    
    # Base model line
    base_acc = results["results"]["base_0_8"]
    latex.append(f"\\addplot[dashed, gray, thick] coordinates {{(2,{base_acc:.2f}) (18,{base_acc:.2f})}};")
    latex.append("\\addlegendentry{Base (0-8 only)}")
    
    latex.append("\\end{axis}")
    latex.append("\\end{tikzpicture}")
    
    return "\n".join(latex)


def generate_full_latex_document(exp1_results: Optional[Dict], exp2_results: Optional[Dict]) -> str:
    """Generate a complete LaTeX document with all tables and plots."""
    latex = []
    latex.append("\\documentclass{article}")
    latex.append("\\usepackage{booktabs}")
    latex.append("\\usepackage{pgfplots}")
    latex.append("\\pgfplotsset{compat=1.18}")
    latex.append("\\usepackage{geometry}")
    latex.append("\\geometry{a4paper, margin=1in}")
    latex.append("")
    latex.append("\\title{RoRA Experiment Results}")
    latex.append("\\author{Generated from Experiment Outputs}")
    latex.append("\\date{\\today}")
    latex.append("")
    latex.append("\\begin{document}")
    latex.append("\\maketitle")
    latex.append("")
    
    if exp1_results:
        latex.append("\\section{Experiment 1: MNIST Multi-Task Learning}")
        latex.append("")
        latex.append(generate_experiment1_table(exp1_results))
        latex.append("")
        latex.append("\\begin{figure}[h]")
        latex.append("\\centering")
        latex.append(generate_experiment1_plot(exp1_results))
        latex.append("\\caption{Performance comparison for multi-task learning}")
        latex.append("\\label{fig:experiment1}")
        latex.append("\\end{figure}")
        latex.append("")
    
    if exp2_results:
        latex.append("\\section{Experiment 2: MNIST Novel Class Adaptation}")
        latex.append("")
        latex.append(generate_experiment2_table(exp2_results))
        latex.append("")
        latex.append("\\begin{figure}[h]")
        latex.append("\\centering")
        latex.append(generate_experiment2_plot(exp2_results))
        latex.append("\\caption{Performance comparison for novel class adaptation}")
        latex.append("\\label{fig:experiment2}")
        latex.append("\\end{figure}")
        latex.append("")
    
    latex.append("\\end{document}")
    
    return "\n".join(latex)


def main():
    """Main function to generate LaTeX output."""
    print("=" * 80)
    print("Generating LaTeX Tables and Plots")
    print("=" * 80)
    print()
    
    # Load results
    exp1_results = load_latest_results("experiment1")
    exp2_results = load_latest_results("experiment2")
    
    if not exp1_results and not exp2_results:
        print("Error: No results found. Please run experiments first.")
        return
    
    # Generate individual components
    output_dir = Path(__file__).parent.parent / "outputs"
    
    if exp1_results:
        print("\nGenerating Experiment 1 LaTeX...")
        table1 = generate_experiment1_table(exp1_results)
        plot1 = generate_experiment1_plot(exp1_results)
        
        with open(output_dir / "experiment1_table.tex", "w") as f:
            f.write(table1)
        print(f"  ✓ Table saved to: {output_dir / 'experiment1_table.tex'}")
        
        with open(output_dir / "experiment1_plot.tex", "w") as f:
            f.write(plot1)
        print(f"  ✓ Plot saved to: {output_dir / 'experiment1_plot.tex'}")
    
    if exp2_results:
        print("\nGenerating Experiment 2 LaTeX...")
        table2 = generate_experiment2_table(exp2_results)
        plot2 = generate_experiment2_plot(exp2_results)
        
        with open(output_dir / "experiment2_table.tex", "w") as f:
            f.write(table2)
        print(f"  ✓ Table saved to: {output_dir / 'experiment2_table.tex'}")
        
        with open(output_dir / "experiment2_plot.tex", "w") as f:
            f.write(plot2)
        print(f"  ✓ Plot saved to: {output_dir / 'experiment2_plot.tex'}")
    
    # Generate full document
    print("\nGenerating full LaTeX document...")
    full_doc = generate_full_latex_document(exp1_results, exp2_results)
    with open(output_dir / "results.tex", "w") as f:
        f.write(full_doc)
    print(f"  ✓ Full document saved to: {output_dir / 'results.tex'}")
    
    print("\n" + "=" * 80)
    print("LaTeX generation complete!")
    print("=" * 80)
    print("\nTo compile the LaTeX document:")
    print("  pdflatex outputs/results.tex")
    print("\nOr use the individual .tex files in your own LaTeX documents.")


if __name__ == "__main__":
    main()
