"""
Visualization dashboard for quantum compression performance.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import os
from datetime import datetime
from ..utils.log_event import log_event
from ..utils.trace import get_trace_history

def plot_compression_metrics(save_path: Optional[str] = None) -> plt.Figure:
    """
    Generate visualization of compression metrics from trace history.

    Args:
        save_path: Path to save the visualization (optional)

    Returns:
        plt.Figure: Visualization figure
    """
    # Get recent trace events related to compression
    traces = get_trace_history(limit=1000)
    compression_traces = [t for t in traces if "compress" in t.get("message", "").lower()]

    if not compression_traces:
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No compression activity found in trace history",
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Extract compression metrics
    timestamps = []
    ratios = []
    sizes = []
    entropies = []

    for trace in compression_traces:
        if "data" in trace and "ratio" in trace["data"]:
            timestamps.append(datetime.fromisoformat(trace.get("timestamp", datetime.now().isoformat())))
            ratios.append(trace["data"].get("ratio", 1.0))
            sizes.append(trace["data"].get("original_size", 0))
            entropies.append(trace["data"].get("entropy", 0.0))

    if not timestamps:
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No compression metrics found in trace history",
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])

    # Compression ratio over time
    ax1 = plt.subplot(gs[0])
    ax1.plot(timestamps, ratios, 'b.-', linewidth=1, alpha=0.7)
    ax1.set_title('Compression Ratio Over Time')
    ax1.set_ylabel('Compression Ratio')
    ax1.set_ylim(0, max(ratios) * 1.1 or 1.0)
    ax1.grid(True, alpha=0.3)

    # Size distribution
    ax2 = plt.subplot(gs[1])
    if sizes:
        ax2.hist(sizes, bins=20, alpha=0.7, color='green')
        ax2.set_title('Document Size Distribution')
        ax2.set_xlabel('Document Size (bytes)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

    # Entropy vs Ratio scatter plot
    ax3 = plt.subplot(gs[2])
    if entropies and ratios:
        ax3.scatter(entropies, ratios, alpha=0.7, c=sizes, cmap='viridis')
        ax3.set_title('Entropy vs Compression Ratio')
        ax3.set_xlabel('Entropy')
        ax3.set_ylabel('Compression Ratio')
        ax3.set_xlim(0, 1)
        ax3.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar.set_label('Document Size')

    plt.tight_layout()

    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            log_event(f"Saved compression metrics visualization to {save_path}")
        except Exception as e:
            log_event(f"Error saving visualization: {str(e)}", level="ERROR")

    return fig

def display_compression_stats(stats: Dict[str, Any]) -> Dict[str, plt.Figure]:
    """
    Generate visualizations of compression statistics.

    Args:
        stats: Compression statistics dictionary

    Returns:
        dict: Dictionary of visualization figures
    """
    figures = {}

    # Summary visualization
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    labels = ['Compressed', 'Uncompressed']
    sizes = [stats.get("documents_compressed", 0), stats.get("documents_uncompressed", 0)]

    if sum(sizes) > 0:
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
               colors=['#4CAF50', '#F44336'])
        ax1.axis('equal')
        ax1.set_title('Document Compression Status')
    else:
        ax1.text(0.5, 0.5, "No compression statistics available",
               ha='center', va='center', fontsize=14)
        ax1.axis('off')

    figures['summary'] = fig1

    # Storage saving visualization
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bytes_saved = stats.get("bytes_saved", 0)
    avg_ratio = stats.get("average_ratio", 0.0)

    if bytes_saved > 0:
        # Create a bar chart showing bytes saved
        ax2.bar(['Original Size', 'Compressed Size'],
               [bytes_saved / (1 - avg_ratio) if avg_ratio < 1 else bytes_saved,
                bytes_saved / avg_ratio if avg_ratio > 0 else 0],
               color=['#2196F3', '#4CAF50'])

        ax2.text(0, bytes_saved / (1 - avg_ratio) * 0.5 if avg_ratio < 1 else bytes_saved * 0.5,
                f"{bytes_saved / 1024:.1f} KB saved",
                ha='center', va='center', rotation=90, color='white', fontweight='bold')

        ax2.set_title(f'Storage Savings (Average Ratio: {avg_ratio:.2f})')
        ax2.set_ylabel('Bytes')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, "No storage savings data available",
               ha='center', va='center', fontsize=14)
        ax2.axis('off')

    figures['savings'] = fig2

    return figures

def get_compression_dashboard(save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """
    Generate a comprehensive compression dashboard.

    Args:
        save_dir: Directory to save visualizations (optional)

    Returns:
        dict: Dictionary of visualization figures
    """
    # Start with metrics from trace
    figures = {"metrics": plot_compression_metrics()}

    # Try to get stats from RAG compression module
    try:
        from ..memory.compression.rag_compression import CompressedRAGStore
        store = CompressedRAGStore()
        stats = store.get_compression_stats()
        stat_figures = display_compression_stats(stats)
        figures.update(stat_figures)
    except (ImportError, AttributeError):
        # Create placeholder figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No compression statistics module available",
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        figures["stats"] = fig

    # Save figures if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figures.items():
            try:
                fig.savefig(os.path.join(save_dir, f"compression_{name}.png"),
                          dpi=300, bbox_inches='tight')
            except Exception as e:
                log_event(f"Error saving {name} visualization: {str(e)}", level="ERROR")

    return figures
