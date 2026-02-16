import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Set
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

logger = logging.getLogger(__name__)

class BaseExporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

class HTMLExporter(BaseExporter):
    def generate_report(self, graph: nx.DiGraph, cycles: List[List[str]],
                       metrics: Dict[str, Any], color_by: str) -> Dict[str, str]:
        template = self._load_template()
        graph_data = self._prepare_graph_data(graph, cycles, metrics)
        
        # Convert sets to lists for JSON serialization
        serializable_metrics = self._make_json_serializable(metrics)
        
        html_content = template.format(
            graph_data=json.dumps(graph_data),
            metrics=json.dumps(serializable_metrics),
            color_by=color_by
        )
        output_path = os.path.join(self.output_dir, "module_report.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return {"path": output_path}
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert sets and other non-serializable objects to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)  # Convert sets to lists
        elif isinstance(obj, tuple):
            return list(obj)  # Convert tuples to lists
        elif hasattr(obj, '__dict__'):
            return str(obj)   # Convert objects to their string representation
        else:
            return obj

    def _load_template(self) -> str:
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'report.html')
        if os.path.exists(template_path):
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                 logger.error(f"Failed to load template file {template_path}: {e}")
        return self._get_default_template()

    def _get_default_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
            <head>
                <title>Module Dependencies Report</title>
                <script src="https://d3js.org/d3.v7.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
                <style>
                    .node {{ cursor: pointer; }}
                    .node:hover {{ stroke-width: 3px; }}
                    .tooltip {{ position: absolute; padding: 10px; background: white; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.1); pointer-events: none; }}
                    .controls {{ position: fixed; top: 10px; left: 10px; background: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px; }}
                    .metrics {{ position: fixed; right: 10px; top: 10px; background: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px; }}
                    svg {{ width: 100%; height: 95vh; border: 1px solid #ccc; }}
                </style>
            </head>
            <body>
                <div class="controls">
                    <button onclick="toggleLayout()">Toggle Layout</button>
                    <button onclick="highlightCycles()">Show Cycles</button>
                    <button onclick="showMetricsPanel()">Show Metrics Panel</button>
                    <select onchange="colorBy(this.value)">
                        <option value="default">Default</option>
                        <option value="complexity">Complexity</option>
                        <option value="community">Communities</option>
                        <option value="centrality">Centrality</option>
                    </select>
                </div>
                <div id="graph"></div>
                <div class="metrics" id="metrics-panel" style="display:none;">Metrics Panel</div>
                <script>
                    const graphData = {{graph_data}};
                    const metrics = {{metrics}};
                    const colorByMode = "{{color_by}}";
                    
                    // Visualization code
                    function toggleLayout() {{ 
                        console.log("Toggle Layout clicked"); 
                    }}
                    
                    function highlightCycles() {{ 
                        console.log("Highlight Cycles clicked"); 
                    }}
                    
                    function showMetricsPanel() {{
                        const panel = document.getElementById('metrics-panel');
                        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
                        console.log("Show Metrics Panel clicked");
                    }}
                    
                    function colorBy(value) {{ 
                        console.log("Color by:", value); 
                    }}

                    // Basic D3 setup
                    const svg = d3.select("#graph").append("svg")
                        .attr("width", "100%")
                        .attr("height", 600);

                    const g = svg.append("g");

                    // Render graph data
                    if (graphData.nodes && graphData.nodes.length > 0) {{
                        const nodes = g.selectAll(".node")
                           .data(graphData.nodes)
                           .enter().append("circle")
                           .attr("class", "node")
                           .attr("r", 5)
                           .attr("cx", function(d, i) {{ return (i % 20) * 30 + 20; }})
                           .attr("cy", function(d, i) {{ return Math.floor(i / 20) * 30 + 20; }})
                           .style("fill", "lightblue");
                    }} else {{
                        g.append("text").text("No graph data to display.").attr("x", 10).attr("y", 20);
                    }}

                </script>
            </body>
        </html>
        """

    def _prepare_graph_data(self, graph: nx.DiGraph, cycles: List[List[str]], metrics: Dict[str, Any]) -> Dict:
        nodes = []
        links = []
        for node in graph.nodes():
            nodes.append({
                'id': node,
                'group': metrics.get('communities', {}).get(node, 0),
                'metrics': {
                    'complexity': metrics.get('complexity', {}).get(node, 0),
                    'centrality': metrics.get('centrality', {}).get('betweenness', {}).get(node, 0),
                    'coupling': metrics.get('coupling', {}).get(node, 0)
                }
            })
        for source, target in graph.edges():
            links.append({
                'source': source,
                'target': target,
                'value': 1
            })
        return {'nodes': nodes, 'links': links}

    def _generate_d3_data(self, graph: nx.DiGraph, orphans: Set[str], entries: Set[str], cycles: List[List[str]]) -> Dict[str, List[Dict[str, Any]]]:
        nodes = []
        cycle_nodes = {node for cycle in cycles for node in cycle} if cycles else set()
        for node, data in graph.nodes(data=True):
            nodes.append({
                "id": node,
                "name": node,
                "size": data.get("size", 1000),
                "path": data.get("path", ""),
                "modified": time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime(data.get("modified", 0))),
                "complexity": data.get("complexity", 0),
                "is_orphan": node in orphans,
                "is_entry": node in entries,
                "is_cycle": node in cycle_nodes
            })
        links = []
        for source, target, data in graph.edges(data=True):
            links.append({
                "source": source,
                "target": target,
                "type": data.get("type", "direct"),
                "count": data.get("count", 1)
            })
        return {"nodes": nodes, "links": links}


    def _get_cycle_report(self, cycles: List[List[str]]) -> str:
        if not cycles:
            return "<p>No circular dependencies detected.</p>"
        report = "<h3>Detected Cycles</h3><ul>"
        for i, cycle in enumerate(cycles):
            if i >= 10: # Limit reported cycles in HTML for brevity
                 report += "<li>... (additional cycles truncated)</li>"
                 break
            report += f"<li>{' -> '.join(f'`{node}`' for node in cycle + [cycle[0]])}</li>" # Add backticks
        report += "</ul>"
        return report

class GraphExporter(BaseExporter):
    def generate_visualization(self, graph: nx.DiGraph, color_by: str,
                             metrics: Dict[str, Any]) -> Optional[Dict[str, str]]:
        if not graph or graph.number_of_nodes() == 0:
             logger.warning("Graph is empty, cannot generate visualization.")
             return None
        try:
            plt.figure(figsize=(max(12, graph.number_of_nodes() / 5), max(8, graph.number_of_nodes() / 8))) # Dynamic size
            pos = nx.spring_layout(graph, k=0.6/np.sqrt(graph.number_of_nodes()) if graph.number_of_nodes() > 0 else 1, seed=42)

            node_sizes = [metrics.get('complexity', {}).get(node, 1) * 50 + 100 for node in graph.nodes()] # Adjusted scaling
            node_colors = self._get_node_colors(graph, color_by, metrics)

            nx.draw(graph, pos, node_size=node_sizes, node_color=node_colors,
                   with_labels=True, arrows=True, font_size=8, alpha=0.8, edge_color='grey')
            output_path = os.path.join(self.output_dir, "module_graph.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Graph visualization saved to {output_path}")
            return {"png_path": output_path}
        except Exception as e:
            logger.error(f"Failed to generate graph visualization: {e}", exc_info=True)
            plt.close() # Ensure plot is closed on error
            return None

    def export_graph(self, graph: nx.DiGraph, format: str = 'json') -> Optional[str]:
        if not graph:
             logger.warning("Graph is empty, cannot export.")
             return None
        output_path = os.path.join(self.output_dir, f"module_graph.{format}")
        try:
            if format == 'json':
                data = nx.node_link_data(graph)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            elif format == 'graphml':
                nx.write_graphml(graph, output_path, encoding='utf-8')
            else:
                logger.error(f"Unsupported graph export format: {format}")
                return None
            logger.info(f"Graph data exported to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to export graph to {output_path}: {e}", exc_info=True)
            return None

    def _get_node_colors(self, graph: nx.DiGraph, color_by: str, metrics: Dict[str, Any]) -> List[Any]:
        num_nodes = graph.number_of_nodes()
        if num_nodes == 0:
            return []

        if color_by == 'complexity':
            return self._get_complexity_colors(graph, metrics)
        elif color_by == 'coupling':
             return self._get_coupling_colors(graph, metrics)
        elif color_by == 'cohesion':
             return self._get_cohesion_colors(graph, metrics)
        elif color_by == 'community':
             return self._get_community_colors(graph, metrics)
        # Add check for orphan/entry/cycle status if metrics contain this info
        elif color_by == 'status' and 'status' in metrics:
             status_map = {'orphan': '#FFFFE0', 'entry': '#90EE90', 'cycle': '#FF6347', 'normal': 'skyblue'}
             return [status_map.get(metrics['status'].get(node, 'normal'), 'grey') for node in graph.nodes()]

        return ['skyblue'] * num_nodes

    def _get_metric_based_colors(self, graph: nx.DiGraph, metrics: Dict[str, Any], metric_key: str, cmap_name: str) -> List[Any]:
        metric_values = metrics.get(metric_key, {})
        if not metric_values:
            return ['grey'] * graph.number_of_nodes()
        
        # Filter out non-numeric values if necessary
        numeric_values = {k: v for k, v in metric_values.items() if isinstance(v, (int, float))}
        if not numeric_values:
             return ['grey'] * graph.number_of_nodes()

        max_value = max(numeric_values.values()) if numeric_values else 1
        max_value = max(1, max_value) # Avoid division by zero
        cmap = plt.cm.get_cmap(cmap_name)

        colors = []
        for node in graph.nodes():
             value = numeric_values.get(node, 0)
             normalized_value = value / max_value
             colors.append(cmap(normalized_value))
        return colors

    def _get_complexity_colors(self, graph: nx.DiGraph, metrics: Dict[str, Any]) -> List[Any]:
        return self._get_metric_based_colors(graph, metrics, 'complexity', 'YlOrRd')

    def _get_coupling_colors(self, graph: nx.DiGraph, metrics: Dict[str, Any]) -> List[Any]:
        return self._get_metric_based_colors(graph, metrics, 'coupling', 'Blues')

    def _get_cohesion_colors(self, graph: nx.DiGraph, metrics: Dict[str, Any]) -> List[Any]:
        return self._get_metric_based_colors(graph, metrics, 'cohesion', 'Greens')

    def _get_community_colors(self, graph: nx.DiGraph, metrics: Dict[str, Any]) -> List[Any]:
        communities = metrics.get('communities', {})
        if not communities:
            return ['grey'] * graph.number_of_nodes()

        num_communities = max(communities.values()) + 1 if communities else 1
        cmap = plt.cm.get_cmap('viridis', num_communities) # Use qualitative map for distinct groups

        colors = []
        for node in graph.nodes():
             community_id = communities.get(node, 0)
             # Normalize id for colormap lookup
             color_val = cmap(community_id / max(1, num_communities -1)) if num_communities > 1 else cmap(0.5)
             colors.append(color_val)
        return colors


    def compare_with_previous(self, graph: nx.DiGraph) -> Optional[Dict[str, Any]]:
        snapshot_path = os.path.join(self.output_dir, "module_snapshot.json")
        if not os.path.exists(snapshot_path):
            logger.info(f"Previous snapshot not found at {snapshot_path}. Cannot compare.")
            return None
        try:
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                previous = json.load(f)

            current = nx.node_link_data(graph) # Assumes graph nodes don't have complex data not suitable for JSON

            prev_nodes_list = previous.get('nodes', [])
            current_nodes_list = current.get('nodes', [])
            prev_links_list = previous.get('links', [])
            current_links_list = current.get('links', [])

            prev_nodes_set = set(item['id'] for item in prev_nodes_list if isinstance(item, dict) and 'id' in item)
            current_nodes_set = set(item['id'] for item in current_nodes_list if isinstance(item, dict) and 'id' in item)

            prev_links_set = set()
            for link in prev_links_list:
                if isinstance(link, dict) and 'source' in link and 'target' in link:
                     # Ensure consistent order for comparison
                     prev_links_set.add(tuple(sorted((link['source'], link['target']))))

            current_links_set = set()
            for link in current_links_list:
                 if isinstance(link, dict) and 'source' in link and 'target' in link:
                     current_links_set.add(tuple(sorted((link['source'], link['target']))))

            comparison = {
                'added_nodes': sorted(list(current_nodes_set - prev_nodes_set)),
                'removed_nodes': sorted(list(prev_nodes_set - current_nodes_set)),
                'added_edges': sorted(list(current_links_set - prev_links_set)),
                'removed_edges': sorted(list(prev_links_set - current_links_set))
            }
            logger.info(f"Comparison complete: {len(comparison['added_nodes'])} added nodes, {len(comparison['removed_nodes'])} removed nodes, {len(comparison['added_edges'])} added edges, {len(comparison['removed_edges'])} removed edges.")
            return comparison

        except (IOError, json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to load or compare snapshot {snapshot_path}: {e}", exc_info=True)
            return None


class TableExporter(BaseExporter):
    def generate_table(self, graph: nx.DiGraph, cycles: List[List[str]],
                      metrics: Dict[str, Any], root_dir: str) -> Optional[Dict[str, str]]:
        if not graph:
            logger.warning("Graph is empty, cannot generate table.")
            return None
        try:
            md_content = self._generate_markdown(graph, cycles, metrics, root_dir)
            output_path = os.path.join(self.output_dir, "module_dependencies.md")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            logger.info(f"Dependency table saved to {output_path}")
            return {"path": output_path}
        except Exception as e:
             logger.error(f"Failed to generate markdown table: {e}", exc_info=True)
             return None

    def _generate_markdown(self, graph: nx.DiGraph, cycles: List[List[str]],
                         metrics: Dict[str, Any], root_dir: str) -> str:
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        content = [
            "# Module Dependencies Report\n",
            f"## Project Structure\n",
            f"- Root: `{root_dir}`\n",
            f"- Modules: {num_nodes}\n",
            f"- Dependencies: {num_edges}\n",
            f"- Cycles: {len(cycles)}\n\n",
            "## Dependencies Table\n",
            "| Module | Depends On | Complexity | Coupling | Cohesion |",
            "|--------|------------|------------|----------|----------|"
        ]
        if num_nodes == 0:
             content.append("| *No modules found* | - | - | - | - |")
        else:
            for node in sorted(graph.nodes()):
                # successors() returns an iterator, convert to list for sorting/joining
                deps = sorted(list(graph.successors(node)))
                deps_str = ', '.join(f'`{d}`' for d in deps) if deps else 'None'

                complexity_val = metrics.get('complexity', {}).get(node)
                coupling_val = metrics.get('coupling', {}).get(node)
                cohesion_val = metrics.get('cohesion', {}).get(node)

                complexity_str = f"{complexity_val:.2f}" if isinstance(complexity_val, (int, float)) else 'N/A'
                coupling_str = f"{coupling_val:.2f}" if isinstance(coupling_val, (int, float)) else 'N/A'
                cohesion_str = f"{cohesion_val:.2f}" if isinstance(cohesion_val, (int, float)) else 'N/A'

                content.append(
                    f"| `{node}` | {deps_str} | "
                    f"{complexity_str} | {coupling_str} | {cohesion_str} |"
                )

        if cycles:
             content.append("\n## Circular Dependencies\n")
             for i, cycle in enumerate(cycles):
                 if i >= 10: # Limit reported cycles
                      content.append("- ... (additional cycles truncated)")
                      break
                 cycle_str = ' -> '.join(f'`{node}`' for node in cycle + [cycle[0]])
                 content.append(f"- {cycle_str}")

        return '\n'.join(content)


class DocumentationExporter(BaseExporter):
    def generate_documentation(self, modules: Dict[str, Dict[str, Any]],
                             metrics: Dict[str, Any], graph: nx.DiGraph) -> Optional[str]:
        if not graph and not modules:
            logger.warning("No modules or graph data provided for documentation.")
            return None
        try:
            doc = {
                'overview': self._generate_overview(modules, graph),
                'architecture': self._generate_architecture_doc(modules, graph),
                'module_docs': self._generate_module_docs(modules),
                'dependencies': self._generate_dependency_docs(graph),
                'metrics': self._generate_metrics_doc(metrics)
            }
            output_path = os.path.join(self.output_dir, "project_documentation.md")

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self._format_documentation(doc))
            logger.info(f"Project documentation saved to {output_path}")
            return output_path
        except IOError as e:
            logger.error(f"Failed to write documentation file {output_path}: {e}")
            return None
        except Exception as e:
             logger.error(f"Failed to generate documentation content: {e}", exc_info=True)
             return None

    def _generate_overview(self, modules: Dict[str, Dict[str, Any]], graph: nx.DiGraph) -> str:
        num_modules = len(modules)
        num_deps = graph.number_of_edges() if graph else 0
        return (f"This project contains {num_modules} modules with {num_deps} internal dependencies.\n"
                f"[Placeholder - Key architectural patterns description requires deeper analysis]\n")

    def _generate_architecture_doc(self, modules: Dict[str, Dict[str, Any]], graph: nx.DiGraph) -> str:
        top_level_modules = sorted(list({m.split('.')[0] for m in modules if '.' in m}))
        root_modules = sorted(list({m for m in modules if '.' not in m}))
        structure_desc = ""
        if top_level_modules:
            structure_desc += f"Top-level packages observed: {', '.join(f'`{pkg}`' for pkg in top_level_modules)}.\n"
        if root_modules:
            structure_desc += f"Root-level modules observed: {', '.join(f'`{mod}`' for mod in root_modules)}.\n"
        if not structure_desc:
             structure_desc = "Project structure appears flat or analysis was limited.\n"

        return (f"{structure_desc}"
                f"[Placeholder - Further architectural details based on graph structure, e.g., layers, component interactions]\n")

    def _generate_module_docs(self, modules: Dict[str, Dict[str, Any]]) -> str:
        lines = ["### Module Details\n"]
        if not modules:
             lines.append("No module details available.")
        else:
            for name, data in sorted(modules.items()):
                path = data.get('path', 'N/A')
                rel_path = os.path.relpath(path) if path != 'N/A' else 'N/A' # Show relative path
                size = data.get('size', 'N/A')
                loc = data.get('loc', 'N/A') # Assuming loc metric exists
                lines.append(f"- **`{name}`**: Path: `{rel_path}`, Size: {size} bytes, LOC: {loc}.")
        return "\n".join(lines)

    def _generate_dependency_docs(self, graph: nx.DiGraph) -> str:
        if not graph or graph.number_of_nodes() == 0:
            return "### Dependency Analysis\nNo dependency data available.\n"

        lines = ["### Dependency Analysis\n"]
        try:
            in_degrees = sorted(graph.in_degree(), key=lambda item: item[1], reverse=True)
            lines.append("**Most Depended-Upon Modules (Top 5):**")
            count = 0
            for node, degree in in_degrees[:5]:
                if degree > 0:
                    lines.append(f"- `{node}` (Imported by {degree} other modules)")
                    count += 1
            if count == 0: lines.append("  (None with significant in-degree)")


            out_degrees = sorted(graph.out_degree(), key=lambda item: item[1], reverse=True)
            lines.append("\n**Modules with Most Outgoing Dependencies (Top 5):**")
            count = 0
            for node, degree in out_degrees[:5]:
                if degree > 0:
                    lines.append(f"- `{node}` (Imports {degree} other modules)")
                    count += 1
            if count == 0: lines.append("  (None with significant out-degree)")

            # Add cycle info if available (pass cycles or re-detect?) Assume cycles are detected elsewhere.
            # cycles = list(nx.simple_cycles(graph)) # Re-detecting might be slow
            # if cycles: ... add cycle summary ...

        except Exception as e:
             lines.append(f"\n*Error during dependency analysis: {e}*")

        return "\n".join(lines)

    def _generate_metrics_doc(self, metrics: Dict[str, Any]) -> str:
        lines = ["### Code Metrics Summary\n"]
        if not metrics:
             lines.append("No metrics data available.")
             return "\n".join(lines)

        complexity = metrics.get('complexity', {})
        if complexity:
            valid_complexities = [v for v in complexity.values() if isinstance(v, (int, float))]
            if valid_complexities:
                 avg_complexity = np.mean(valid_complexities)
                 max_complexity_node = max(complexity, key=lambda k: complexity.get(k, 0) if isinstance(complexity.get(k), (int, float)) else -1)
                 lines.append(f"- Average Complexity Score: {avg_complexity:.2f}")
                 lines.append(f"- Highest Complexity Module: `{max_complexity_node}` ({complexity[max_complexity_node]})")
            else:
                 lines.append("- Complexity metrics not available or not numeric.")
        else:
            lines.append("- Complexity metrics not calculated.")

        coupling = metrics.get('coupling', {})
        if coupling:
             valid_couplings = [v for v in coupling.values() if isinstance(v, (int, float))]
             if valid_couplings:
                 avg_coupling = np.mean(valid_couplings)
                 lines.append(f"- Average Coupling Score: {avg_coupling:.2f}")
             else:
                 lines.append("- Coupling metrics not available or not numeric.")

        return "\n".join(lines)

    def _format_documentation(self, doc: Dict[str, str]) -> str:
        formatted_doc = f"# Project Documentation\n\n"
        formatted_doc += f"## Overview\n{doc.get('overview', 'Not available.')}\n"
        formatted_doc += f"## Architecture\n{doc.get('architecture', 'Not available.')}\n"
        formatted_doc += f"## Module Details\n{doc.get('module_docs', 'Not available.')}\n"
        formatted_doc += f"## Dependencies\n{doc.get('dependencies', 'Not available.')}\n"
        formatted_doc += f"## Metrics\n{doc.get('metrics', 'Not available.')}\n"
        return formatted_doc

class AnimationExporter(BaseExporter):
    def generate_evolution_animation(self, snapshots: List[nx.DiGraph]) -> Optional[str]:
        if not snapshots:
            logger.warning("No snapshots provided for animation.")
            return None
        if len(snapshots) < 2:
            logger.warning("Need at least two snapshots to generate an evolution animation.")
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        combined_graph = nx.DiGraph()
        all_nodes = set()
        valid_snapshots = []
        for i, g in enumerate(snapshots):
             if not isinstance(g, nx.DiGraph):
                 logger.warning(f"Item at index {i} in snapshots is not a NetworkX DiGraph. Skipping.")
                 continue
             valid_snapshots.append(g)
             all_nodes.update(g.nodes())
             combined_graph.add_nodes_from(g.nodes())
             combined_graph.add_edges_from(g.edges())

        if not valid_snapshots:
             logger.warning("No valid DiGraph objects found in snapshots list.")
             plt.close(fig)
             return None
        if not all_nodes:
             logger.warning("Snapshots contain no nodes. Cannot generate animation.")
             plt.close(fig)
             return None

        try:
             pos = nx.spring_layout(combined_graph, k=0.6/np.sqrt(len(all_nodes)) if len(all_nodes)>0 else 1, seed=42)
        except Exception as e:
             logger.warning(f"Layout calculation failed for combined graph: {e}. Using random layout.")
             pos = nx.random_layout(combined_graph, seed=42)

        output_path = os.path.join(self.output_dir, "dependency_evolution.gif")

        num_frames = len(valid_snapshots)

        def animate(i):
            ax.clear()
            if i >= num_frames:
                 return
            graph = valid_snapshots[i]

            current_nodes = set(graph.nodes())
            current_pos = {node: p for node, p in pos.items() if node in current_nodes}

            if not current_pos:
                 ax.text(0.5, 0.5, f"Snapshot {i+1}: Empty Graph", ha='center', va='center')
                 ax.set_title(f"Dependency Evolution - Step {i+1}/{num_frames}")
                 ax.axis('off')
                 return

            nx.draw_networkx_nodes(graph, current_pos, ax=ax, node_color='lightblue', node_size=400, alpha=0.8)
            nx.draw_networkx_edges(graph, current_pos, ax=ax, edge_color='grey', alpha=0.5, arrows=True, arrowsize=10)
            nx.draw_networkx_labels(graph, current_pos, ax=ax, font_size=7)

            ax.set_title(f"Dependency Evolution - Step {i+1}/{num_frames}")
            ax.axis('off')

        try:
            anim = animation.FuncAnimation(fig, animate,
                                         frames=num_frames,
                                         interval=1500,
                                         repeat=False)

            anim.save(output_path, writer='pillow', fps=0.66)
            logger.info(f"Dependency evolution animation saved to {output_path}")
            plt.close(fig)
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate or save animation to {output_path}: {e}", exc_info=True)
            plt.close(fig)
            return None