"""
Visualization components for the MetaConsciousness SDK.

This module provides visualization tools for monitoring and analyzing
metacognitive processes.
"""
from .dashboard import (
    generate_dashboard,
    plot_meta_state,
    plot_performance_history,
    render_thought_tree,
    render_reasoning_chain,
    plot_tool_usage,
    plot_rag_activity,
    plot_confidence_heatmap,
    generate_thought_dashboard
)

from .compression_monitor import (
    plot_compression_metrics,
    display_compression_stats,
    get_compression_dashboard
)

__all__ = [
    'generate_dashboard',
    'plot_meta_state',
    'plot_performance_history',
    'render_thought_tree',
    'render_reasoning_chain',
    'plot_tool_usage',
    'plot_rag_activity',
    'plot_confidence_heatmap',
    'generate_thought_dashboard',
    'plot_compression_metrics',
    'display_compression_stats',
    'get_compression_dashboard'
]
