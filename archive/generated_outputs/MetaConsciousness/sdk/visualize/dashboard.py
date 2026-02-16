"""
Enhanced visualization dashboard for MetaConsciousness.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import os
from datetime import datetime

# Fix imports to use absolute paths
from MetaConsciousness.utils.log_event import log_event
from MetaConsciousness.utils.trace import get_trace_history
from MetaConsciousness.agent.agent_tools.tool_executor import default_executor
from MetaConsciousness.rag.engine import default_rag_engine
from MetaConsciousness.memory.vectorstore import VectorStore

def generate_dashboard(meta_instance, save_path: Optional[str] = None) -> List[plt.Figure]:
    """
    Generate a complete visualization dashboard.

    Args:
        meta_instance: MetaConsciousness instance
        save_path: Path to save figures (optional)

    Returns:
        list: Generated figures
    """
    figures = []

    # Awareness and regulation history
    fig_history = plot_meta_state(meta_instance, save_path)
    figures.append(fig_history)

    # Performance metrics
    if hasattr(meta_instance, 'performance_history') and meta_instance.performance_history:
        fig_performance = plot_performance_history(meta_instance, save_path)
        figures.append(fig_performance)

    # Think engine visualization if available
    if hasattr(meta_instance, 'thought_engine'):
        # Tree visualization
        if hasattr(meta_instance.thought_engine, 'branches') and meta_instance.thought_engine.branches:
            fig_tree = render_thought_tree(meta_instance.thought_engine, save_path)
            figures.append(fig_tree)

        # Chain visualization
        if hasattr(meta_instance.thought_engine, 'chain') and meta_instance.thought_engine.chain:
            fig_chain = render_reasoning_chain(meta_instance.thought_engine.chain, save_path)
            figures.append(fig_chain)

    # Tool usage visualization
    try:
        if hasattr(default_executor, 'execution_history') and default_executor.execution_history:
            fig_tools = plot_tool_usage(default_executor.execution_history, save_path)
            figures.append(fig_tools)
    except ImportError:
        pass

    # RAG visualization
    try:
        if hasattr(default_rag_engine, 'vector_store') and default_rag_engine.vector_store:
            fig_rag = plot_rag_activity(default_rag_engine, save_path)
            figures.append(fig_rag)
    except ImportError:
        pass

    return figures

def plot_meta_state(meta_instance, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot awareness and regulation history.

    Args:
        meta_instance: MetaConsciousness instance
        save_path: Path to save the figure (optional)

    Returns:
        plt.Figure: The generated figure
    """
    # Extract history data
    awareness_history = meta_instance.awareness_history
    regulation_history = meta_instance.regulation_history

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot awareness
    x_vals = range(len(awareness_history))
    ax1.plot(x_vals, awareness_history, 'b-', linewidth=2, label='Awareness')
    ax1.set_title('Metaconsciousness Awareness History')
    ax1.set_ylabel('Awareness Level')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot regulation
    ax2.plot(x_vals, regulation_history, 'r-', linewidth=2, label='Regulation')
    ax2.set_title('Metaconsciousness Regulation History')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Regulation Level')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    # Save figure if requested
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(f"{save_path}_meta_state.png", dpi=300, bbox_inches='tight')

    return fig

def plot_performance_history(meta_instance, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot performance history metrics.

    Args:
        meta_instance: MetaConsciousness instance
        save_path: Path to save the figure (optional)

    Returns:
        plt.Figure: The generated figure
    """
    # Extract performance history
    performance_history = np.array(meta_instance.performance_history)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each performance metric
    x_vals = range(len(performance_history))
    for i in range(performance_history.shape[1]):
        metric_name = f"Metric {i+1}"
        ax.plot(x_vals, performance_history[:, i], marker='o', linestyle='-',
                alpha=0.7, label=metric_name)

    ax.set_title('Performance Metrics History')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Metric Value')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save figure if requested
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(f"{save_path}_performance.png", dpi=300, bbox_inches='tight')

    return fig

def render_thought_tree(thought_engine, save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize tree of thought structure.

    Args:
        thought_engine: ThoughtEngine instance or data
        save_path: Path to save the figure (optional)

    Returns:
        plt.Figure: The generated figure
    """
    # Extract branches
    if hasattr(thought_engine, 'branches'):
        branches = thought_engine.branches
    elif isinstance(thought_engine, dict) and 'branches' in thought_engine:
        branches = thought_engine['branches']
    else:
        branches = thought_engine

    # Create graph
    G = nx.DiGraph()

    # Add root node
    G.add_node("root", label="Root", type="root")

    # Process each branch
    for branch_id, branch in branches.items():
        # Add branch node
        G.add_node(branch_id,
                  label=f"Branch {branch_id[-4:]}",
                  confidence=getattr(branch, 'confidence', 0.5),
                  evaluation=getattr(branch, 'evaluation_score', 0.5),
                  type="branch")

        # Connect to root or parent
        parent_id = getattr(branch, 'parent_id', None) or "root"
        if parent_id not in G:
            parent_id = "root"
        G.add_edge(parent_id, branch_id)

        # Process steps in branch
        steps = getattr(branch, 'steps', [])
        prev_node = branch_id

        for i, step in enumerate(steps):
            step_id = getattr(step, 'step_id', f"{branch_id}_step_{i}")
            step_text = getattr(step, 'reasoning', str(step))
            if len(step_text) > 30:
                step_text = step_text[:27] + "..."

            # Add step node
            G.add_node(step_id,
                      label=step_text,
                      confidence=getattr(step, 'confidence', 0.5),
                      type="step")

            # Connect to previous node
            G.add_edge(prev_node, step_id)
            prev_node = step_id

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate node positions using hierarchical layout
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB')

    # Draw nodes with different colors based on type
    node_colors = []
    node_sizes = []

    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'step')
        if node_type == 'root':
            node_colors.append('lightblue')
            node_sizes.append(500)
        elif node_type == 'branch':
            # Color based on evaluation score
            eval_score = G.nodes[node].get('evaluation', 0.5)
            color = plt.cm.RdYlGn(eval_score)
            node_colors.append(color)
            node_sizes.append(300)
        else:  # step
            # Color based on confidence
            confidence = G.nodes[node].get('confidence', 0.5)
            color = plt.cm.Blues(confidence)
            node_colors.append(color)
            node_sizes.append(200)

    # Draw the graph
    nx.draw_networkx(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=8,
        arrows=True,
        arrowsize=15,
        width=1.5,
        labels={node: G.nodes[node]['label'] for node in G.nodes()},
        edge_color='gray',
        alpha=0.8
    )

    ax.set_title("Tree of Thought Visualization")
    ax.axis('off')

    # Create legend
    root_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                           markersize=10, label='Root')
    branch_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                             markersize=10, label='Branch')
    step_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                           markersize=10, label='Step')
    ax.legend(handles=[root_patch, branch_patch, step_patch], loc='upper right')

    plt.tight_layout()

    # Save figure if requested
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(f"{save_path}_thought_tree.png", dpi=300, bbox_inches='tight')

    return fig

def render_reasoning_chain(chain_steps, save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize chain of thought structure.

    Args:
        chain_steps: List of ChainStep objects or data
        save_path: Path to save the figure (optional)

    Returns:
        plt.Figure: The generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Number of steps
    n_steps = len(chain_steps)
    if n_steps == 0:
        ax.text(0.5, 0.5, "No chain steps available", ha='center', va='center')
        ax.axis('off')
        return fig

    # Create chain layout
    step_width = 0.8
    step_height = 0.15
    x_positions = np.linspace(0, 1, n_steps)
    y_position = 0.5

    # Add step boxes
    for i, step in enumerate(chain_steps):
        # Extract step data
        step_id = getattr(step, 'step_id', f"step_{i}")
        reasoning = getattr(step, 'reasoning', str(step))
        confidence = getattr(step, 'confidence', 0.5)

        # Truncate long reasoning text
        if len(reasoning) > 50:
            reasoning = reasoning[:47] + "..."

        # Calculate box position
        x = x_positions[i]
        y = y_position

        # Draw box with confidence-based color
        box_color = plt.cm.Blues(confidence)
        rect = plt.Rectangle((x - step_width/2, y - step_height/2),
                           step_width, step_height,
                           color=box_color, alpha=0.7)
        ax.add_patch(rect)

        # Add step text
        ax.text(x, y, f"Step {i+1}\n{reasoning}",
                ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.3'))

        # Add arrow to next step
        if i < n_steps - 1:
            arrow_len = x_positions[i+1] - x_positions[i]
            ax.arrow(x + step_width/2 - 0.05, y,
                    arrow_len - step_width + 0.07, 0,
                    head_width=0.02, head_length=0.02,
                    fc='black', ec='black')

    ax.set_title("Chain of Thought Visualization")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()

    # Save figure if requested
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(f"{save_path}_reasoning_chain.png", dpi=300, bbox_inches='tight')

    return fig

def plot_tool_usage(execution_history, save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize tool usage patterns.

    Args:
        execution_history: Tool execution history
        save_path: Path to save the figure (optional)

    Returns:
        plt.Figure: The generated figure
    """
    if not execution_history:
        # Create empty figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No tool usage data available", ha='center', va='center')
        ax.axis('off')
        return fig

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # Tool frequency subplot
    ax1 = plt.subplot(gs[0])

    # Count tool usage frequency
    tool_counts = {}
    success_counts = {}
    failure_counts = {}

    for record in execution_history:
        tool_name = record.get('tool_name', 'unknown')
        success = record.get('success', False)

        # Update counts
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        if success:
            success_counts[tool_name] = success_counts.get(tool_name, 0) + 1
        else:
            failure_counts[tool_name] = failure_counts.get(tool_name, 0) + 1

    # Plot tool usage counts
    tools = list(tool_counts.keys())
    success_vals = [success_counts.get(tool, 0) for tool in tools]
    failure_vals = [failure_counts.get(tool, 0) for tool in tools]

    x = np.arange(len(tools))
    width = 0.35

    ax1.bar(x - width/2, success_vals, width, label='Success', color='green', alpha=0.7)
    ax1.bar(x + width/2, failure_vals, width, label='Failure', color='red', alpha=0.7)

    ax1.set_title('Tool Usage Frequency')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tools, rotation=45, ha='right')
    ax1.set_ylabel('Usage Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Recent executions timeline subplot
    ax2 = plt.subplot(gs[1])

    # Get most recent executions (up to 10)
    recent_executions = execution_history[-10:] if len(execution_history) > 10 else execution_history

    # Plot executions
    y_positions = np.arange(len(recent_executions))
    tool_names = [record.get('tool_name', 'unknown') for record in recent_executions]
    success_flags = [record.get('success', False) for record in recent_executions]

    # Color based on success
    colors = ['green' if success else 'red' for success in success_flags]

    ax2.barh(y_positions, [1] * len(recent_executions), color=colors, alpha=0.7)

    # Add result text
    for i, record in enumerate(recent_executions):
        tool_name = record.get('tool_name', 'unknown')
        args = record.get('arguments', {})
        result = record.get('result', '')

        # Format arguments
        args_text = ', '.join(f"{k}={v}" for k, v in args.items())

        # Truncate result if too long
        if len(result) > 30:
            result = result[:27] + "..."

        # Add text
        ax2.text(0.01, y_positions[i], f"{tool_name}({args_text}) → {result}",
                va='center', fontsize=8)

    ax2.set_title('Recent Tool Executions')
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([f"{i+1}" for i in range(len(recent_executions))])
    ax2.set_ylabel('Execution Index')
    ax2.set_xlim(0, 1)
    ax2.set_xticks([])

    plt.tight_layout()

    # Save figure if requested
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(f"{save_path}_tool_usage.png", dpi=300, bbox_inches='tight')

    return fig

def plot_rag_activity(rag_engine, save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize RAG system activity.

    Args:
        rag_engine: RAG engine instance
        save_path: Path to save the figure (optional)

    Returns:
        plt.Figure: The generated figure
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Check if vector store has documents
    vector_store = getattr(rag_engine, 'vector_store', None)
    document_count = vector_store.count() if vector_store else 0

    # No documents case
    if document_count == 0:
        ax1.text(0.5, 0.5, "No documents in vector store", ha='center', va='center')
        ax2.text(0.5, 0.5, "No retrieval activity to display", ha='center', va='center')
        ax1.axis('off')
        ax2.axis('off')
        return fig

    # Plot document store size
    ax1.bar(0, document_count, width=0.5, color='blue', alpha=0.7)
    ax1.set_title('Vector Store Document Count')
    ax1.set_xticks([])
    ax1.set_ylabel('Document Count')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot retrieval activities from trace history
    traces = get_trace_history(limit=100)

    # Filter RAG-related traces
    rag_traces = [trace for trace in traces
                 if trace.get('action', '').startswith('rag_') or
                 'retrieve' in trace.get('action', '').lower()]

    if not rag_traces:
        ax2.text(0.5, 0.5, "No RAG activity found in trace history", ha='center', va='center')
        ax2.axis('off')
    else:
        # Extract activity timestamps and types
        timestamps = []
        activity_types = []
        retrieval_counts = []

        for trace in rag_traces:
            action = trace.get('action', '')
            timestamp = trace.get('timestamp', '')
            data = trace.get('data', {})

            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    timestamps.append(dt)
                    activity_types.append(action)

                    # Try to extract document count
                    count = data.get('document_count', 0)
                    if count == 0 and 'documents' in data:
                        count = len(data['documents'])
                    retrieval_counts.append(count)
                except ValueError:
                    # Skip invalid timestamps
                    continue

        if timestamps:
            # Convert to relative time (seconds ago)
            now = datetime.now()
            rel_times = [(now - dt).total_seconds() / 60 for dt in timestamps]  # minutes ago

            # Plot activity timeline
            ax2.scatter(rel_times, retrieval_counts, c=np.arange(len(rel_times)),
                       cmap='viridis', s=100, alpha=0.7)

            # Connect points
            ax2.plot(rel_times, retrieval_counts, 'gray', alpha=0.3)

            # Customize plot
            ax2.set_title('RAG Retrieval Activity')
            ax2.set_xlabel('Minutes Ago')
            ax2.set_ylabel('Docs Retrieved')
            ax2.grid(True, alpha=0.3)

            # Invert x-axis so most recent is on the right
            ax2.invert_xaxis()
        else:
            ax2.text(0.5, 0.5, "No timestamp data for RAG activity", ha='center', va='center')
            ax2.axis('off')

    plt.tight_layout()

    # Save figure if requested
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(f"{save_path}_rag_activity.png", dpi=300, bbox_inches='tight')

    return fig

def plot_confidence_heatmap(thought_data, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a confidence heatmap visualization.

    Args:
        thought_data: Thought data (engine or dictionary)
        save_path: Path to save the figure (optional)

    Returns:
        plt.Figure: The generated figure
    """
    # Extract chain steps
    chain_steps = []
    if hasattr(thought_data, 'chain'):
        chain_steps = thought_data.chain
    elif isinstance(thought_data, dict) and 'chain' in thought_data:
        chain_steps = thought_data['chain']

    # Extract branches
    branches = {}
    if hasattr(thought_data, 'branches'):
        branches = thought_data.branches
    elif isinstance(thought_data, dict) and 'branches' in thought_data:
        branches = thought_data['branches']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # No thought data case
    if not chain_steps and not branches:
        ax.text(0.5, 0.5, "No thought data available", ha='center', va='center')
        ax.axis('off')
        return fig

    # Prepare data matrix
    max_steps = 0
    branch_data = []
    branch_names = []

    # Process chain first (if exists)
    if chain_steps:
        branch_names.append("Main Chain")
        chain_conf = [getattr(step, 'confidence', 0.5) for step in chain_steps]
        branch_data.append(chain_conf)
        max_steps = max(max_steps, len(chain_conf))

    # Process branches
    for branch_id, branch in branches.items():
        branch_names.append(f"Branch {branch_id[-6:]}")

        steps = getattr(branch, 'steps', [])
        step_conf = []

        for step in steps:
            if hasattr(step, 'confidence'):
                step_conf.append(step.confidence)
            elif isinstance(step, dict) and 'confidence' in step:
                step_conf.append(step['confidence'])
            else:
                step_conf.append(0.5)  # Default confidence

        branch_data.append(step_conf)
        max_steps = max(max_steps, len(step_conf))

    # Pad all branches to same length
    for i in range(len(branch_data)):
        branch_len = len(branch_data[i])
        if branch_len < max_steps:
            branch_data[i] = branch_data[i] + [np.nan] * (max_steps - branch_len)

    # Convert to numpy array
    data_matrix = np.array(branch_data)

    # Create heatmap
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')

    # Customize plot
    ax.set_title('Thought Confidence Heatmap')
    ax.set_xlabel('Step Number')
    ax.set_ylabel('Branch')

    # Set tick labels
    ax.set_yticks(np.arange(len(branch_names)))
    ax.set_yticklabels(branch_names)
    ax.set_xticks(np.arange(max_steps))
    ax.set_xticklabels([f"{i+1}" for i in range(max_steps)])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Confidence')

    # Add confidence values as text
    for i in range(len(branch_names)):
        for j in range(max_steps):
            if j < len(branch_data[i]) and not np.isnan(data_matrix[i, j]):
                ax.text(j, i, f"{data_matrix[i, j]:.2f}",
                       ha="center", va="center",
                       color="black" if 0.3 <= data_matrix[i, j] <= 0.7 else "white",
                       fontsize=8)

    plt.tight_layout()

    # Save figure if requested
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(f"{save_path}_confidence_heatmap.png", dpi=300, bbox_inches='tight')

    return fig

def generate_thought_dashboard(thought_engine, save_path: Optional[str] = None) -> List[plt.Figure]:
    """
    Generate a complete thought visualization dashboard.

    Args:
        thought_engine: ThoughtEngine instance
        save_path: Path to save figures (optional)

    Returns:
        list: Generated figures
    """
    figures = []

    # Tree visualization
    fig_tree = render_thought_tree(thought_engine, save_path)
    figures.append(fig_tree)

    # Chain visualization
    fig_chain = render_reasoning_chain(thought_engine.chain, save_path)
    figures.append(fig_chain)

    # Confidence heatmap
    fig_heatmap = plot_confidence_heatmap(thought_engine, save_path)
    figures.append(fig_heatmap)

    return figures

def render_thought_tree(thought_engine, save_path=None) -> None:
    """
    Render the thought tree as a graph.

    Args:
        thought_engine: ThoughtEngine instance
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    # Create a graph
    G = nx.DiGraph()

    # Check if we have branches or steps
    has_nodes = False

    # Add branches
    for branch_id, branch in thought_engine.branches.items():
        G.add_node(branch_id, label=f"Branch: {branch_id}",
                   type="branch", confidence=branch.confidence)
        has_nodes = True

        # Add steps within the branch
        for i, step in enumerate(branch.steps):
            step_id = f"{branch_id}_step_{i}"
            G.add_node(step_id, label=step[:30] + "..." if len(step) > 30 else step,
                       type="step")
            G.add_edge(branch_id, step_id)

    # Add chain steps
    for i, step in enumerate(thought_engine.chain):
        G.add_node(step.id, label=step.reasoning[:30] + "..." if len(step.reasoning) > 30 else step.reasoning,
                   type="step", confidence=step.confidence)
        has_nodes = True

        # Connect sequential steps
        if i > 0:
            G.add_edge(thought_engine.chain[i-1].id, step.id)

    # If no nodes, create a blank graph
    if not has_nodes:
        G.add_node("no_thoughts", label="No thoughts recorded")

    # Create the figure
    plt.figure(figsize=(10, 8))

    try:
        # Try to use graphviz for layout (better but requires external dependency)
        try:
            import pygraphviz
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB')
        except (ImportError, OSError) as e:
            # Fallback to built-in layout if graphviz not available
            pos = nx.spring_layout(G, seed=42)

        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                font_size=8, node_size=2000, alpha=0.8,
                edge_color='gray', arrows=True)

        # Save if path provided
        if save_path:
            plt.savefig(save_path)

        return plt.gcf()

    except Exception as e:
        # Create a basic error figure as fallback
        plt.clf()
        plt.figtext(0.5, 0.5, f"Visualization Error: {str(e)}",
                    ha='center', va='center', fontsize=12)

        # Save if path provided
        if save_path:
            plt.savefig(save_path)

        return plt.gcf()

def generate_thought_dashboard(data, save_path=None) -> None:
    """
    Generate an HTML dashboard visualizing the thought process.

    Args:
        data: Dashboard data
        save_path: Optional path to save the dashboard

    Returns:
        HTML string of the dashboard
    """
    try:
        # Convert data to a JSON-friendly format for display
        import json
        displayable_data = {}

        # Extract display-friendly data
        if "art_controller" in data:
            displayable_data["art_controller"] = {
                "vigilance": data["art_controller"].get("vigilance", "N/A"),
                "categories": data["art_controller"].get("categories", {})
            }

        if "meta_core" in data:
            displayable_data["meta_core"] = {
                "awareness": data["meta_core"].get("awareness", "N/A"),
                "regulation": data["meta_core"].get("regulation", "N/A"),
                "meta_state": data["meta_core"].get("meta_state", "N/A")
            }

        if "performance" in data:
            displayable_data["performance"] = data["performance"]

        if "reflex" in data:
            displayable_data["reflex"] = data["reflex"]

        # Create basic HTML dashboard
        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>MetaConsciousness Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ margin-bottom: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .data-section {{ margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                .meta-stats {{ display: flex; justify-content: space-around; }}
                .stat-box {{ text-align: center; padding: 10px; background-color: #e9f7fe; border-radius: 5px; min-width: 100px; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow: auto; }}
                .visualization-error {{ color: #721c24; background-color: #f8d7da; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>MetaConsciousness System Dashboard</h1>
                    <p>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="data-section">
                    <h2>Metacognitive State</h2>
                    <div class="meta-stats">
                        <div class="stat-box">
                            <h3>Awareness</h3>
                            <p>{displayable_data.get("meta_core", {}).get("awareness", "N/A")}</p>
                        </div>
                        <div class="stat-box">
                            <h3>Regulation</h3>
                            <p>{displayable_data.get("meta_core", {}).get("regulation", "N/A")}</p>
                        </div>
                        <div class="stat-box">
                            <h3>Meta State</h3>
                            <p>{displayable_data.get("meta_core", {}).get("meta_state", "N/A")}</p>
                        </div>
                        <div class="stat-box">
                            <h3>Vigilance</h3>
                            <p>{displayable_data.get("art_controller", {}).get("vigilance", "N/A")}</p>
                        </div>
                    </div>
                </div>

                <div class="data-section">
                    <h2>Visualization</h2>
                    <div class="visualization-error">
                        <p>Interactive visualization requires Graphviz installation. Displaying data in text format instead.</p>
                    </div>
                </div>

                <div class="data-section">
                    <h2>System Data</h2>
                    <pre>{json.dumps(displayable_data, indent=2, default=str)}</pre>
                </div>
            </div>
        </body>
        </html>
        """

        return html
    except Exception as e:
        # Fallback to extremely simple HTML in case of any errors
        return f"""<!DOCTYPE html>
        <html>
        <head><title>MetaConsciousness Dashboard (Error Recovery Mode)</title></head>
        <body>
            <h1>MetaConsciousness Dashboard</h1>
            <p>Dashboard generation encountered an error: {str(e)}</p>
            <p>System is still operational. This is a visualization-only error.</p>
        </body>
        </html>
        """

# ...existing code...
