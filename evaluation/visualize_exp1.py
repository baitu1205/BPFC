"""
Experiment 1 Visualization Module

This module provides visualization tools for the Local vs. Global Behavior Profile experiment.
It creates plots and tables to visualize:
1. Behavior profile changes across rounds
2. ASR vs BTF relationship
3. Delta analysis for each region
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PlotConfig:
    """Configuration for plotting."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 150
    style: str = 'seaborn-v0_8-darkgrid'
    color_palette: str = 'Set2'
    font_size: int = 12
    title_font_size: int = 14
    save_format: str = 'png'


class ExperimentVisualizer:
    """
    Visualizer for Experiment 1 results.
    """

    def __init__(self, results_path: str, output_dir: str):
        """
        Initialize the visualizer.

        Args:
            results_path: Path to the behavior_profile_results.json file
            output_dir: Directory to save plots
        """
        self.results_path = results_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load results
        with open(results_path, 'r') as f:
            self.data = json.load(f)

        self.results = self.data.get('results', [])
        print(f"Loaded {len(self.results)} evaluation results")

    def plot_behavior_profile_over_time(
        self,
        save: bool = True,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot behavior profiles (Local vs Global) over rounds.

        Shows how p_T, p_E, p_B, p_N change for both Local and Global models.
        """
        if not self.results:
            print("No results to plot!")
            return None

        # Extract data
        rounds = []
        local_T, local_E, local_B, local_N = [], [], [], []
        global_T, global_E, global_B, global_N = [], [], [], []

        for r in self.results:
            rounds.append(r['round'])
            local = r['local']
            global_ = r['global']
            local_T.append(local['p_T'])
            local_E.append(local['p_E'])
            local_B.append(local['p_B'])
            local_N.append(local['p_N'])
            global_T.append(global_['p_T'])
            global_E.append(global_['p_E'])
            global_B.append(global_['p_B'])
            global_N.append(global_['p_N'])

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        regions = [
            ('Target', local_T, global_T, axes[0, 0]),
            ('Equivalent', local_E, global_E, axes[0, 1]),
            ('Boundary', local_B, global_B, axes[1, 0]),
            ('Normal', local_N, global_N, axes[1, 1]),
        ]

        for region_name, local_data, global_data, ax in regions:
            ax.plot(rounds, local_data, 'o-', label='Local', linewidth=2, markersize=6)
            ax.plot(rounds, global_data, 's--', label='Global', linewidth=2, markersize=6)
            ax.set_xlabel('Round', fontsize=11)
            ax.set_ylabel('Refusal Rate', fontsize=11)
            ax.set_title(f'{region_name} Region', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

        fig.suptitle('Behavior Profile: Local vs Global Over Rounds', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'behavior_profile_over_time.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")

        if show:
            plt.show()

        return fig

    def plot_delta_analysis(
        self,
        save: bool = True,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot the delta (change) for each region over rounds.

        Key insight: Large delta_B indicates behavior boundary shift.
        """
        if not self.results:
            return None

        rounds = []
        delta_T, delta_E, delta_B, delta_N = [], [], [], []

        for r in self.results:
            rounds.append(r['round'])
            delta = r['delta']
            delta_T.append(delta['T'])
            delta_E.append(delta['E'])
            delta_B.append(delta['B'])
            delta_N.append(delta['N'])

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(rounds, delta_T, 'o-', label='Δ Target', linewidth=2, markersize=6)
        ax.plot(rounds, delta_E, 's-', label='Δ Equivalent', linewidth=2, markersize=6)
        ax.plot(rounds, delta_B, '^-', label='Δ Boundary', linewidth=2, markersize=8, color='red')
        ax.plot(rounds, delta_N, 'd-', label='Δ Normal', linewidth=2, markersize=6)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.fill_between(rounds, 0, delta_B, alpha=0.2, color='red', label='Boundary Change')

        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Δ (Global - Local)', fontsize=12)
        ax.set_title('Behavior Change After FedAvg Aggregation', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'delta_analysis.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")

        if show:
            plt.show()

        return fig

    def plot_btf_over_time(
        self,
        save: bool = True,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot BTF (Behavior Transferability Factor) over rounds.

        Lower BTF = more behavior change = potential attack leakage
        """
        if not self.results:
            return None

        rounds = []
        btf_values = []

        for r in self.results:
            rounds.append(r['round'])
            btf_values.append(r['btf'])

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(rounds, btf_values, 'o-', linewidth=2, markersize=8, color='purple')
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label='Perfect BTF (no change)')
        ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, label='Medium BTF')

        # Fill regions
        ax.fill_between(rounds, btf_values, 1.0, alpha=0.2, color='green')
        ax.fill_between(rounds, btf_values, 0.0, where=[b < 0.5 for b in btf_values], alpha=0.2, color='red')

        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('BTF', fontsize=12)
        ax.set_title('Behavior Transferability Factor (BTF) Over Rounds', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'btf_over_time.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")

        if show:
            plt.show()

        return fig

    def plot_asr_vs_btf(
        self,
        save: bool = True,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot ASR (Attack Success Rate) vs BTF.

        The key phenomenon: High ASR + Low BTF
        """
        if not self.results:
            return None

        # Calculate ASR as average of Target and Equivalent regions (Global)
        asr_values = []
        btf_values = []

        for r in self.results:
            global_ = r['global']
            asr = (global_['p_T'] + global_['p_E']) / 2  # Average ASR
            asr_values.append(asr)
            btf_values.append(r['btf'])

        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot
        scatter = ax.scatter(btf_values, asr_values, c=range(len(asr_values)),
                            cmap='viridis', s=100, alpha=0.7, edgecolors='black')

        # Add colorbar to show round progression
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Round', fontsize=11)

        # Add quadrant lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

        # Highlight the target quadrant (High ASR, Low BTF)
        ax.fill_between([0, 0.5], 0.5, 1.0, alpha=0.1, color='red')
        ax.text(0.25, 0.75, 'High ASR\nLow BTF\n(Target)', fontsize=12, ha='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('BTF (Behavior Transferability Factor)', fontsize=12)
        ax.set_ylabel('ASR (Attack Success Rate)', fontsize=12)
        ax.set_title('ASR vs BTF: The Key Phenomenon\n(High ASR + Low BTF)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'asr_vs_btf.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")

        if show:
            plt.show()

        return fig

    def create_summary_table(self) -> str:
        """
        Create a summary table of results.

        Returns:
            Markdown formatted table
        """
        if not self.results:
            return "No results available"

        # Get first, middle, and last results
        indices = [0, len(self.results)//2, len(self.results)-1]
        selected = [self.results[i] for i in indices if i < len(self.results)]

        table = "| Round | Client | Local (T/E/B/N) | Global (T/E/B/N) | ΔB | BTF |\n"
        table += "|-------|--------|----------------|-----------------|-----|------|\n"

        for r in selected:
            local = f"{r['local']['p_T']:.2f}/{r['local']['p_E']:.2f}/{r['local']['p_B']:.2f}/{r['local']['p_N']:.2f}"
            global_ = f"{r['global']['p_T']:.2f}/{r['global']['p_E']:.2f}/{r['global']['p_B']:.2f}/{r['global']['p_N']:.2f}"
            delta_b = f"{r['delta']['B']:+.2f}"
            btf = f"{r['btf']:.3f}"

            table += f"| {r['round']} | {r['client']} | {local} | {global_} | {delta_b} | {btf} |\n"

        return table

    def print_analysis(self):
        """
        Print analysis of the key phenomenon.
        """
        if not self.results:
            print("No results to analyze!")
            return

        print("\n" + "=" * 70)
        print("EXPERIMENT 1 ANALYSIS: Local vs Global Behavior Profile")
        print("=" * 70)

        # Key metrics
        first = self.results[0]
        last = self.results[-1]

        print("\n1. OVERALL TRENDS:")
        print("-" * 40)

        # Calculate average ASR and BTF
        avg_asr_local = np.mean([(r['local']['p_T'] + r['local']['p_E'])/2 for r in self.results])
        avg_asr_global = np.mean([(r['global']['p_T'] + r['global']['p_E'])/2 for r in self.results])
        avg_btf = np.mean([r['btf'] for r in self.results])
        avg_delta_b = np.mean([r['delta']['B'] for r in self.results])

        print(f"   Average Local ASR:  {avg_asr_local:.2%}")
        print(f"   Average Global ASR: {avg_asr_global:.2%}")
        print(f"   Average BTF: {avg_btf:.4f}")
        print(f"   Average Δ Boundary: {avg_delta_b:+.2%}")

        print("\n2. KEY PHENOMENON CHECK:")
        print("-" * 40)

        # Check for High ASR + Low BTF
        high_asr = avg_asr_global > 0.5
        low_btf = avg_btf < 0.8

        print(f"   High ASR (>50%):  {'✓' if high_asr else '✗'} ({avg_asr_global:.2%})")
        print(f"   Low BTF (<0.8):   {'✓' if low_btf else '✗'} ({avg_btf:.4f})")

        if high_asr and low_btf:
            print("\n   ★ PHENOMENON DETECTED: High ASR + Low BTF")
            print("     The attack remains effective but behavior boundary has shifted!")
        else:
            print("\n   The expected phenomenon was not strongly observed.")

        print("\n3. BEHAVIOR BOUNDARY ANALYSIS:")
        print("-" * 40)

        # Check if boundary changed significantly
        boundary_shift = abs(avg_delta_b) > 0.1
        print(f"   Significant boundary shift detected: {'✓' if boundary_shift else '✗'} ({avg_delta_b:+.2%})")

        if avg_delta_b > 0:
            print("   → Boundary region shows INCREASED refusal rate after aggregation")
            print("   → The backdoor behavior is 'leaking' to similar but non-target prompts")
        elif avg_delta_b < 0:
            print("   → Boundary region shows DECREASED refusal rate after aggregation")
            print("   → The backdoor behavior is being diluted")

        print("\n4. NORMAL REGION STABILITY:")
        print("-" * 40)

        avg_delta_n = np.mean([r['delta']['N'] for r in self.results])
        print(f"   Δ Normal: {avg_delta_n:+.2%}")
        print(f"   Normal behavior stable: {'✓' if abs(avg_delta_n) < 0.1 else '✗'}")

        print("\n" + "=" * 70)

    def generate_all_plots(self):
        """Generate all visualization plots."""
        print("\nGenerating visualizations...")

        self.plot_behavior_profile_over_time(show=False)
        self.plot_delta_analysis(show=False)
        self.plot_btf_over_time(show=False)
        self.plot_asr_vs_btf(show=False)

        print("\nAll plots saved to:", self.output_dir)


def visualize_experiment_results(
    results_path: str,
    output_dir: str = None
):
    """
    Main function to visualize experiment results.

    Args:
        results_path: Path to behavior_profile_results.json
        output_dir: Directory to save plots (defaults to same dir as results)
    """
    if output_dir is None:
        output_dir = os.path.dirname(results_path) or './'

    visualizer = ExperimentVisualizer(results_path, output_dir)
    visualizer.generate_all_plots()
    visualizer.print_analysis()

    print("\n" + visualizer.create_summary_table())


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Visualize Experiment 1 Results')
    parser.add_argument('--results_path', type=str, required=True,
                       help='Path to behavior_profile_results.json')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots')

    args = parser.parse_args()

    visualize_experiment_results(args.results_path, args.output_dir)
