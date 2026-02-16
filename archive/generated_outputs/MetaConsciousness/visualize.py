"""
Visualization tools for plotting awareness, regulation, and trends.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Union
from .utils import calculate_moving_average, ensure_directory
import os
from datetime import datetime

def plot_meta_state(history, save_path=None, show=True, window_size=5) -> None:
    """
    Plot awareness and regulation history.

    Args:
        history: Dictionary with 'awareness_history' and 'regulation_history'
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
        window_size: Window size for moving average

    Returns:
        tuple: Figure and axes objects
    """
    # Extract history data
    awareness_history = history.get('awareness_history', [])
    regulation_history = history.get('regulation_history', [])

    if not awareness_history:
        print("No awareness history to plot")
        return None, None

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot awareness
    x = range(len(awareness_history))
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Awareness', color='tab:blue')
    ax1.plot(x, awareness_history, 'b-', alpha=0.5, label='Awareness')

    # Add moving average for awareness
    if len(awareness_history) >= window_size:
        awareness_ma = calculate_moving_average(awareness_history, window_size)
        ax1.plot(x, awareness_ma, 'b-', label=f'Awareness (MA-{window_size})')

    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1.1)

    # Add regulation on secondary y-axis if available
    if regulation_history:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Regulation', color='tab:red')
        ax2.plot(x, regulation_history, 'r-', alpha=0.5, label='Regulation')

        # Add moving average for regulation
        if len(regulation_history) >= window_size:
            regulation_ma = calculate_moving_average(regulation_history, window_size)
            ax2.plot(x, regulation_ma, 'r-', label=f'Regulation (MA-{window_size})')

        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(0, 1.1)

    # Add title and legend
    plt.title('Meta-State History: Awareness and Regulation')
    lines1, labels1 = ax1.get_legend_handles_labels()

    if regulation_history:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')

    plt.grid(True, alpha=0.3)
    fig.tight_layout()

    # Save if path provided
    if save_path:
        ensure_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax1

def plot_regulation_curve(history, save_path=None, show=True) -> None:
    """
    Plot regulation curve: awareness vs. regulation.

    Args:
        history: Dictionary with 'awareness_history' and 'regulation_history'
        save_path: Path to save the plot (optional)
        show: Whether to display the plot

    Returns:
        tuple: Figure and axes objects
    """
    # Extract history data
    awareness_history = history.get('awareness_history', [])
    regulation_history = history.get('regulation_history', [])

    if not awareness_history or not regulation_history:
        print("Insufficient history to plot regulation curve")
        return None, None

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatterplot of awareness vs. regulation
    scatter = ax.scatter(
        awareness_history,
        regulation_history,
        c=range(len(awareness_history)),  # Color by time
        cmap='viridis',
        alpha=0.7,
        s=50
    )

    # Add colorbar to show time progression
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time Steps')

    # Add connecting lines to show transitions
    ax.plot(awareness_history, regulation_history, 'k-', alpha=0.3)

    # Add arrows to show direction
    if len(awareness_history) > 1:
        for i in range(1, len(awareness_history), max(1, len(awareness_history) // 20)):
            ax.annotate(
                "",
                xy=(awareness_history[i], regulation_history[i]),
                xytext=(awareness_history[i-1], regulation_history[i-1]),
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.6)
            )

    # Mark start and end points
    if awareness_history:
        ax.plot(awareness_history[0], regulation_history[0], 'go', markersize=10, label='Start')
        ax.plot(awareness_history[-1], regulation_history[-1], 'ro', markersize=10, label='Current')

    # Set axis labels and title
    ax.set_xlabel('Awareness')
    ax.set_ylabel('Regulation')
    ax.set_title('Meta-State Regulation Curve')

    # Set axis limits with padding
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Save if path provided
    if save_path:
        ensure_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Regulation curve saved to {save_path}")

    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax

def plot_performance_vs_awareness(performance_history, awareness_history, save_path=None, show=True) -> None:
    """
    Plot performance versus awareness.

    Args:
        performance_history: List of performance values
        awareness_history: List of awareness values
        save_path: Path to save the plot (optional)
        show: Whether to display the plot

    Returns:
        tuple: Figure and axes objects
    """
    if not performance_history or not awareness_history:
        print("Insufficient history to plot performance vs. awareness")
        return None, None

    # Ensure both histories have the same length
    min_len = min(len(performance_history), len(awareness_history))
    performance = [np.mean(p) if isinstance(p, (list, np.ndarray)) else p
                   for p in performance_history[:min_len]]
    awareness = awareness_history[:min_len]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot performance and awareness
    x = range(min_len)
    ax.plot(x, performance, 'b-', label='Performance')
    ax.plot(x, awareness, 'r-', label='Awareness')

    # Add correlation coefficient
    correlation = np.corrcoef(performance, awareness)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
            transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7))

    # Set axis labels and title
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value')
    ax.set_title('Performance vs. Awareness')

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Save if path provided
    if save_path:
        ensure_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax

def generate_dashboard(meta_consciousness, save_dir=None, show=True) -> None:
    """
    Generate a comprehensive dashboard with multiple plots.

    Args:
        meta_consciousness: MetaConsciousness instance
        save_dir: Directory to save plots (optional)
        show: Whether to display the plots

    Returns:
        dict: Dictionary of generated figure objects
    """
    # Extract history data
    history = {
        'awareness_history': meta_consciousness.awareness_history,
        'regulation_history': meta_consciousness.regulation_history,
        'performance_history': meta_consciousness.performance_history
    }

    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    figures = {}

    # Generate meta-state plot
    if save_dir:
        save_path = os.path.join(save_dir, f"meta_state_{timestamp}.png")
    else:
        save_path = None

    fig1, _ = plot_meta_state(history, save_path, show)
    figures['meta_state'] = fig1

    # Generate regulation curve
    if save_dir:
        save_path = os.path.join(save_dir, f"regulation_curve_{timestamp}.png")
    else:
        save_path = None

    fig2, _ = plot_regulation_curve(history, save_path, show)
    figures['regulation_curve'] = fig2

    # Generate performance vs. awareness plot
    if history['performance_history']:
        if save_dir:
            save_path = os.path.join(save_dir, f"performance_vs_awareness_{timestamp}.png")
        else:
            save_path = None

        fig3, _ = plot_performance_vs_awareness(
            history['performance_history'],
            history['awareness_history'],
            save_path,
            show
        )
        figures['performance_vs_awareness'] = fig3

    return figures

def render_thought_tree(tree_data: Dict[str, Any], save_path: Optional[str] = None,
                      show: bool = True) -> tuple:
    """
    Render a visualization of a thought tree.

    Args:
        tree_data: Dictionary of branch data with steps
        save_path: Path to save the visualization (optional)
        show: Whether to display the plot

    Returns:
        tuple: Figure and axes objects
    """
    # Create directed graph
    G = nx.DiGraph()

    # Extract branches and their steps
    branches = tree_data.get("branches", {})

    # Add nodes for branches
    for branch_id, branch in branches.items():
        G.add_node(branch_id,
                  type="branch",
                  confidence=branch.get("confidence", 0.0),
                  score=branch.get("evaluation_score", 0.0))

        # Add nodes for steps in this branch
        steps = branch.get("steps", [])
        for i, step in enumerate(steps):
            # Handle both dict and object representations
            step_id = step.get("step_id", f"{branch_id}_step_{i}") if isinstance(step, dict) else step.step_id

            # Add step node
            G.add_node(step_id,
                      type="step",
                      confidence=step.get("confidence", 0.0) if isinstance(step, dict) else step.confidence)

            # Connect step to branch
            G.add_edge(branch_id, step_id)

            # Connect steps in sequence
            if i > 0:
                prev_step_id = steps[i-1].get("step_id") if isinstance(steps[i-1], dict) else steps[i-1].step_id
                G.add_edge(prev_step_id, step_id)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define positions using hierarchical layout
    pos = nx.spring_layout(G)

    # Draw nodes with different colors for branches and steps
    branch_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "branch"]
    step_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "step"]

    # Get confidence values for node colors
    branch_confidence = [G.nodes[n].get("confidence", 0.5) for n in branch_nodes]
    step_confidence = [G.nodes[n].get("confidence", 0.5) for n in step_nodes]

    # Draw branch nodes
    nx.draw_networkx_nodes(G, pos, nodelist=branch_nodes,
                          node_color=branch_confidence,
                          cmap=plt.cm.viridis,
                          node_size=800,
                          alpha=0.8)

    # Draw step nodes
    nx.draw_networkx_nodes(G, pos, nodelist=step_nodes,
                          node_color=step_confidence,
                          cmap=plt.cm.plasma,
                          node_size=500,
                          node_shape='s',
                          alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos,
                          arrows=True,
                          arrowstyle='->',
                          arrowsize=15)

    # Add labels
    node_labels = {n: n.split('_')[0] + '...' for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # Add title and legends
    plt.title('Thought Tree Visualization')
    plt.axis('off')

    # Add colorbar for confidence
    sm_branch = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm_branch.set_array([])
    cbar_branch = plt.colorbar(sm_branch, ax=ax, shrink=0.6, pad=0.05, orientation='vertical')
    cbar_branch.set_label('Branch Confidence')

    sm_step = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
    sm_step.set_array([])
    cbar_step = plt.colorbar(sm_step, ax=ax, shrink=0.6, pad=0.07, orientation='vertical')
    cbar_step.set_label('Step Confidence')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        ensure_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Thought tree visualization saved to {save_path}")

    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax

def render_reasoning_chain(chain_data: Dict[str, Any], save_path: Optional[str] = None,
                        show: bool = True) -> tuple:
    """
    Render a visualization of a reasoning chain.

    Args:
        chain_data: Dictionary of chain step data
        save_path: Path to save the visualization (optional)
        show: Whether to display the plot

    Returns:
        tuple: Figure and axes objects
    """
    # Extract steps
    steps = chain_data.get("steps", {})

    # Sort steps by timestamp if available
    sorted_steps = sorted(steps.values(),
                         key=lambda x: x.get("timestamp", "") if isinstance(x, dict) else x.timestamp)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract step confidences and IDs
    confidences = []
    step_ids = []

    for step in sorted_steps:
        if isinstance(step, dict):
            confidences.append(step.get("confidence", 0.5))
            step_ids.append(step.get("step_id", "unknown"))
        else:
            confidences.append(step.confidence)
            step_ids.append(step.step_id)

    # Plot the chain as a sequence of connected points
    x = list(range(len(confidences)))
    ax.plot(x, confidences, 'b-', marker='o', markersize=10, linewidth=2)

    # Add confidence value labels
    for i, conf in enumerate(confidences):
        ax.text(i, conf + 0.05, f"{conf:.2f}", ha='center')

    # Add step ID labels
    ax.set_xticks(x)
    ax.set_xticklabels([s[:8] + '...' for s in step_ids], rotation=45, ha='right')

    # Add labels and title
    ax.set_xlabel('Reasoning Steps')
    ax.set_ylabel('Confidence')
    ax.set_title('Chain of Thought Reasoning')

    # Set y-axis limits
    ax.set_ylim(-0.05, 1.05)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        ensure_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reasoning chain visualization saved to {save_path}")

    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax

def plot_confidence_heatmap(structured_thought_data: Dict[str, Any],
                          save_path: Optional[str] = None,
                          show: bool = True) -> tuple:
    """
    Plot a confidence heatmap for thought structures.

    Args:
        structured_thought_data: Dictionary of thought branches and steps
        save_path: Path to save the visualization (optional)
        show: Whether to display the plot

    Returns:
        tuple: Figure and axes objects
    """
    # Extract branch data
    branches = structured_thought_data.get("branches", {})

    if not branches:
        print("No branch data to visualize")
        return None, None

    # Create a matrix of confidence values
    # Rows = branches, Columns = step positions
    branch_ids = list(branches.keys())
    max_steps = max(len(branch.get("steps", [])) for branch in branches.values())

    # Initialize confidence matrix with NaN values
    confidence_matrix = np.full((len(branch_ids), max_steps), np.nan)

    # Fill in confidence values
    for i, branch_id in enumerate(branch_ids):
        branch = branches[branch_id]
        steps = branch.get("steps", [])

        for j, step in enumerate(steps):
            if isinstance(step, dict):
                confidence_matrix[i, j] = step.get("confidence", 0.5)
            else:
                confidence_matrix[i, j] = step.confidence

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap with masked NaN values
    masked_matrix = np.ma.masked_invalid(confidence_matrix)
    cmap = plt.cm.viridis
    cmap.set_bad('white', 1.0)

    heatmap = ax.matshow(masked_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Confidence')

    # Add labels
    ax.set_xlabel('Step Position')
    ax.set_ylabel('Branch')
    ax.set_title('Thought Confidence Heatmap')

    # Set tick labels
    ax.set_xticks(range(max_steps))
    ax.set_xticklabels([f"Step {i+1}" for i in range(max_steps)])

    ax.set_yticks(range(len(branch_ids)))
    ax.set_yticklabels([b[:10] + '...' if len(b) > 10 else b for b in branch_ids])

    plt.tight_layout()

    # Save if path provided
    if save_path:
        ensure_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence heatmap saved to {save_path}")

    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax

def generate_thought_dashboard(thought_data: Dict[str, Any], save_dir: Optional[str] = None,
                             show: bool = True) -> Dict[str, Any]:
    """
    Generate a comprehensive dashboard of thought visualizations.

    Args:
        thought_data: Thought data including branches and chains
        save_dir: Directory to save plots (optional)
        show: Whether to display the plots

    Returns:
        dict: Dictionary of generated figure objects
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figures = {}

    # Generate thought tree visualization
    if save_dir:
        save_path = os.path.join(save_dir, f"thought_tree_{timestamp}.png")
    else:
        save_path = None

    fig1, _ = render_thought_tree(thought_data, save_path, show)
    figures['thought_tree'] = fig1

    # Generate reasoning chain visualization
    if save_dir:
        save_path = os.path.join(save_dir, f"reasoning_chain_{timestamp}.png")
    else:
        save_path = None

    fig2, _ = render_reasoning_chain(thought_data, save_path, show)
    figures['reasoning_chain'] = fig2

    # Generate confidence heatmap
    if save_dir:
        save_path = os.path.join(save_dir, f"confidence_heatmap_{timestamp}.png")
    else:
        save_path = None

    fig3, _ = plot_confidence_heatmap(thought_data, save_path, show)
    figures['confidence_heatmap'] = fig3

    return figures
