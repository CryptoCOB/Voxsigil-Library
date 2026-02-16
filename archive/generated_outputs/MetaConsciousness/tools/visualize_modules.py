import os
import sys
import argparse
import time
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Any, Optional, Set
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _is_excluded(path: str, root_dir: str, exclude_patterns: List[str]) -> bool:
    relative_path = os.path.relpath(path, root_dir)
    path_parts = relative_path.split(os.path.sep)

    for pattern in exclude_patterns:
        if any(part == pattern for part in path_parts):
            logger.debug(f"Excluding '{relative_path}' due to pattern '{pattern}' (direct match)")
            return True
        if pattern.startswith("*") and relative_path.endswith(pattern[1:]):
             logger.debug(f"Excluding '{relative_path}' due to pattern '{pattern}' (suffix match)")
             return True
        if pattern.endswith("*") and relative_path.startswith(pattern[:-1]):
             logger.debug(f"Excluding '{relative_path}' due to pattern '{pattern}' (prefix match)")
             return True
        if pattern in path_parts:
             logger.debug(f"Excluding '{relative_path}' due to pattern '{pattern}' (path part match)")
             return True
    return False

def _get_module_name_from_path(file_path: str, root_dir: str) -> str:
    rel_path = os.path.relpath(file_path, root_dir)
    if os.path.basename(rel_path) == "__init__.py":
        module_path = os.path.dirname(rel_path).replace(os.path.sep, ".")
    else:
        module_path = rel_path.replace(os.path.sep, ".").replace(".py", "")
    if module_path == ".":
        module_path = os.path.basename(root_dir).replace(".py", "")
    if module_path.startswith('.'):
        module_path = module_path[1:]
    return module_path

def _parse_import_line(line: str) -> List[Tuple[str, Optional[str]]]:
    imports = []
    line = line.strip()
    if line.startswith('#'):
        return imports

    parts = line.split('#', 1)
    line = parts[0].strip()

    if line.startswith('import '):
        modules_str = line[7:].strip()
        modules = [m.strip() for m in modules_str.split(',')]
        for module in modules:
            if module:
                base_module = module.split('.')[0]
                imports.append((base_module, None))

    elif line.startswith('from '):
        parts = line.split(' import ')
        if len(parts) >= 2:
            module_str = parts[0][5:].strip()
            if module_str.startswith('.'):
                base_module = module_str.lstrip('.').split('.')[0]
                if not base_module:
                    logger.debug(f"Skipping relative import line: {line}")
                else:
                    imported_items = [item.strip() for item in parts[1].split(',')]
                    for item in imported_items:
                         if item:
                            imports.append((base_module, item))
            else:
                 base_module = module_str.split('.')[0]
                 imported_items = [item.strip() for item in parts[1].split(',')]
                 for item in imported_items:
                     if item:
                        imports.append((base_module, item))

    return imports

def _calculate_node_size(module_data: Dict[str, Any], min_size: int = 500, max_size: int = 5000, scale_factor: float = 100.0) -> float:
    raw_size = module_data.get("size", 0)
    if raw_size > 0:
        scaled_size = np.log1p(raw_size) * scale_factor
    else:
        scaled_size = min_size
    return max(min_size, min(max_size, scaled_size))

def _calculate_node_color(module_data: Dict[str, Any], color_by: str = 'default', is_orphan: bool = False, is_entry: bool = False, is_cycle: bool = False) -> str:
    if is_cycle:
        return '#FF6347'
    if is_orphan:
        return '#FFFFE0'
    if is_entry:
        return '#90EE90'

    color = 'skyblue'

    if color_by == 'complexity':
        complexity = module_data.get('complexity', 0)
        if complexity > 50: color = '#FF4500'
        elif complexity > 20: color = '#FFA500'
        elif complexity > 5: color = '#FFD700'
    elif color_by == 'modified':
        mod_time = module_data.get('modified', 0)
        age_days = (time.time() - mod_time) / (60 * 60 * 24) if mod_time > 0 else float('inf')
        if age_days < 7: color = '#ADD8E6'
        elif age_days < 30: color = '#87CEEB'
        else: color = '#B0C4DE'
    elif color_by == 'size':
        size = module_data.get('size', 0)
        if size > 100000: color = '#DC143C'
        elif size > 20000: color = '#FF69B4'
        elif size > 5000: color = '#DB7093'

    return color

def _find_python_files(directory: str, root_dir: str, exclude_patterns: List[str]) -> List[str]:
    python_files = []
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs[:] = [d for d in dirs if not _is_excluded(os.path.join(root, d), root_dir, exclude_patterns)]

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if not _is_excluded(file_path, root_dir, exclude_patterns):
                    python_files.append(file_path)
                else:
                    logger.debug(f"Skipping excluded file: {file_path}")

    return python_files

def _save_json_data(data: Any, filepath: str) -> bool:
    try:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.debug(f"Successfully saved JSON data to {filepath}")
        return True
    except (IOError, TypeError) as e:
        logger.error(f"Error saving JSON data to {filepath}: {e}")
        return False

def _load_json_data(filepath: str) -> Optional[Any]:
    if not os.path.exists(filepath):
        logger.debug(f"JSON file not found: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded JSON data from {filepath}")
        return data
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error loading JSON data from {filepath}: {e}")
        return None

def _generate_d3_data(graph: nx.DiGraph, orphans: Set[str], entries: Set[str], cycles: List[List[str]]) -> Dict[str, List[Dict[str, Any]]]:
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

def _generate_html_template(graph_data: Dict[str, List[Dict[str, Any]]], stats: Dict[str, Any], cycles: Optional[List[List[str]]] = None) -> str:
    cycle_report = "<h3>Detected Cycles</h3><ul>" + "".join(f"<li>{' -> '.join(cycle + [cycle[0]])}</li>" for cycle in cycles) + "</ul>" if cycles else "<p>No circular dependencies detected.</p>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Module Dependency Visualization</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; overflow: hidden; }}
            #graph-container {{ position: relative; width: 100vw; height: 100vh; }}
            #graph {{ width: 100%; height: 100%; }}
            .node circle {{ stroke: #fff; stroke-width: 1.5px; cursor: pointer; }}
            .link {{ stroke: #999; stroke-opacity: 0.6; stroke-width: 1.5px; }}
            .link.highlighted {{ stroke: #ff3333; stroke-opacity: 1; stroke-width: 2.5px; }}
            .node text {{ pointer-events: none; font-size: 10px; fill: #333; text-anchor: middle; }}
            .node text.hover {{ font-weight: bold; }}
            #tooltip {{ position: absolute; background-color: rgba(255, 255, 255, 0.95); border: 1px solid #ccc; border-radius: 4px; padding: 10px; display: none; box-shadow: 0 2px 5px rgba(0,0,0,0.2); font-size: 12px; max-width: 300px; pointer-events: none; }}
            #controls {{ position: absolute; top: 10px; left: 10px; background-color: rgba(255, 255, 255, 0.9); padding: 15px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); max-height: 90vh; overflow-y: auto; }}
            #controls h2 {{ margin-top: 0; }}
            #controls label, #controls input, #controls button {{ margin-bottom: 5px; display: block; }}
            #controls input[type=text], #controls select {{ width: 90%; padding: 5px; }}
            #controls button {{ padding: 5px 10px; margin-top: 10px; }}
            #cycle-report {{ margin-top: 15px; font-size: 0.9em; }}
            .node.orphan circle {{ fill: #FFFFE0 !important; stroke: #ccc; }}
            .node.entry circle {{ fill: #90EE90 !important; stroke: #55a555; }}
            .node.cycle circle {{ fill: #FF6347 !important; stroke: #b22222; }}
            .legend {{ list-style: none; padding-left: 0; font-size: 0.8em; }}
            .legend li span {{ display: inline-block; width: 12px; height: 12px; margin-right: 5px; border: 1px solid #ccc; vertical-align: middle; }}
        </style>
        <script src="https://d3js.org/d3.v7.min.js"></script>
    </head>
    <body>
        <div id="controls">
            <h2>Module Dependencies</h2>
            <p>Nodes: {stats['node_count']} | Edges: {stats['edge_count']}</p>
            <p>Orphans: {stats['orphan_count']} | Entries: {stats['entry_count']} | Cycles: {stats['cycle_count']}</p>
            <div>
                <label for="search">Search Module:</label>
                <input type="text" id="search" placeholder="Start typing...">
            </div>
            <div>
                <label for="color-mode">Color Nodes By:</label>
                <select id="color-mode">
                    <option value="default" selected>Type (Default)</option>
                    <option value="complexity">Complexity</option>
                    <option value="modified">Last Modified</option>
                    <option value="size">File Size</option>
                </select>
            </div>
            <button id="zoomIn">Zoom In (+)</button>
            <button id="zoomOut">Zoom Out (-)</button>
            <button id="resetZoom">Reset View</button>
             <button id="centerGraph">Center Graph</button>
            <div id="legend-container" style="margin-top:15px;">
                <h4>Legend</h4>
                <ul class="legend">
                    <li><span style="background-color: skyblue;"></span> Standard Module</li>
                    <li><span style="background-color: #FFFFE0;"></span> Orphan Module</li>
                    <li><span style="background-color: #90EE90;"></span> Entry Point</li>
                    <li><span style="background-color: #FF6347;"></span> Part of Cycle</li>
                    <li>Color modes affect standard modules.</li>
                </ul>
            </div>
             <div id="cycle-report">
                {cycle_report}
            </div>
        </div>
        <div id="tooltip"></div>
        <div id="graph-container">
            <svg id="graph"></svg>
        </div>

        <script>
        const graphData = {json.dumps(graph_data, indent=2)};
        const container = d3.select("#graph-container");
        const svg = d3.select("#graph");
        const tooltip = d3.select("#tooltip");
        let width = container.node().getBoundingClientRect().width;
        let height = container.node().getBoundingClientRect().height;

        svg.attr("width", width).attr("height", height);

        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", zoomed);

        const g = svg.append("g");
        svg.call(zoom);

        const initialTransform = d3.zoomIdentity.translate(width / 2, height / 2).scale(0.6);
        svg.call(zoom.transform, initialTransform);
        g.attr("transform", initialTransform);

        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(d => 100 + Math.sqrt(d.source.size + d.target.size)/5).strength(0.5))
            .force("charge", d3.forceManyBody().strength(-600))
            .force("center", d3.forceCenter(0, 0))
            .force("collision", d3.forceCollide().radius(d => Math.sqrt(d.size) / 15 + 15));

        const link = g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(graphData.links)
            .join("line")
            .attr("class", "link")
            .attr("stroke-width", d => 1 + Math.log1p(d.count || 1) / 2)
             .attr("marker-end", "url(#arrowhead)");

        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 8)
            .attr("markerHeight", 8)
            .attr("xoverflow", "visible")
            .append("svg:path")
            .attr("d", "M 0,-5 L 10 ,0 L 0,5")
            .attr("fill", "#999")
            .style("stroke", "none");

        const nodeGroup = g.append("g")
            .attr("class", "nodes")
            .selectAll(".node")
            .data(graphData.nodes)
            .join("g")
            .attr("class", d => `node ${{d.is_orphan ? 'orphan' : ''}} ${{d.is_entry ? 'entry' : ''}} ${{d.is_cycle ? 'cycle' : ''}}`)
            .call(drag(simulation));

        const nodeCircles = nodeGroup.append("circle")
            .attr("r", d => Math.max(8, Math.sqrt(d.size) / 15 + 5))
            .attr("fill", d => getNodeColor(d, 'default'));

        const nodeLabels = nodeGroup.append("text")
            .text(d => d.name.length > 25 ? d.name.substring(0, 22) + "..." : d.name)
            .attr("font-size", "10px");

        nodeGroup.on("mouseover", function(event, d) {{
            const nodeElement = d3.select(this);
            nodeElement.select('circle').transition().duration(100).attr('r', parseFloat(nodeElement.select('circle').attr('r')) + 3);
            nodeElement.select('text').classed('hover', true);

            tooltip.style("display", "block")
                .html(`<strong>${{d.name}}</strong><br>
                      ${{d.path ? 'Path: ' + d.path + '<br>' : ''}}
                      Size: ${{d.size}} bytes<br>
                      Modified: ${{d.modified}}<br>
                      Complexity (Lines/Func/Class): ${{d.complexity || 'N/A'}}`)
                .style("left", (event.pageX + 15) + "px")
                .style("top", (event.pageY - 10) + "px");

            link.classed('highlighted', l => l.source.id === d.id || l.target.id === d.id);
            nodeGroup.style("opacity", n => (hasConnection(d.id, n.id) || n.id === d.id) ? 1 : 0.3);

        }}).on("mouseout", function(event, d) {{
            const nodeElement = d3.select(this);
            nodeElement.select('circle').transition().duration(100).attr('r', Math.max(8, Math.sqrt(d.size) / 15 + 5));
             nodeElement.select('text').classed('hover', false);

            tooltip.style("display", "none");
            link.classed('highlighted', false);
            nodeGroup.style("opacity", 1);
        }});

        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            nodeGroup.attr("transform", d => `translate(${{d.x}},${{d.y}})`);

             d3.select("#arrowhead").attr("refX", (d) => {{
                const targetNode = graphData.nodes.find(n => n.id === d.target.id);
                return targetNode ? Math.max(8, Math.sqrt(targetNode.size) / 15 + 5) + 8 : 20;
            }});

        }});

        function zoomed(event) {{
            g.attr("transform", event.transform);
        }}

        const linkSet = new Set(graphData.links.map(l => `${{l.source.id}}->${{l.target.id}}`));
        const reverseLinkSet = new Set(graphData.links.map(l => `${{l.target.id}}->${{l.source.id}}`));
        function hasConnection(sourceId, targetId) {{
            return linkSet.has(`${{sourceId}}->${{targetId}}`) || linkSet.has(`${{targetId}}->${{sourceId}}`) || reverseLinkSet.has(`${{sourceId}}->${{targetId}}`) || reverseLinkSet.has(`${{targetId}}->${{sourceId}}`);
        }}

        function drag(simulation) {{
            function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}
            function dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}
            function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
            }}
            return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
        }}

        d3.select("#zoomIn").on("click", () => svg.transition().call(zoom.scaleBy, 1.3));
        d3.select("#zoomOut").on("click", () => svg.transition().call(zoom.scaleBy, 1 / 1.3));
        d3.select("#resetZoom").on("click", () => svg.transition().duration(750).call(zoom.transform, initialTransform));
        d3.select("#centerGraph").on("click", () => {{
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            nodeGroup.each(function(d) {{
                minX = Math.min(minX, d.x); maxX = Math.max(maxX, d.x);
                minY = Math.min(minY, d.y); maxY = Math.max(maxY, d.y);
            }});
            if (minX === Infinity) return;

            const currentTransform = d3.zoomTransform(svg.node());
            const graphWidth = (maxX - minX) * currentTransform.k;
            const graphHeight = (maxY - minY) * currentTransform.k;
            const scale = Math.min(0.9 * width / graphWidth, 0.9 * height / graphHeight, 2);

            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;

             const targetX = width / 2 - centerX * scale;
             const targetY = height / 2 - centerY * scale;

            const newTransform = d3.zoomIdentity.translate(targetX, targetY).scale(scale);
            svg.transition().duration(750).call(zoom.transform, newTransform);
        }});

        d3.select("#search").on("input", function() {{
            const term = this.value.toLowerCase();
            nodeGroup.style("opacity", d => term === "" || d.name.toLowerCase().includes(term) ? 1 : 0.1);
            link.style("opacity", d => term === "" || d.source.name.toLowerCase().includes(term) || d.target.name.toLowerCase().includes(term) ? 0.6 : 0.05);
        }});

        d3.select("#color-mode").on("change", function() {{
            const mode = this.value;
            nodeCircles.transition().duration(500).attr("fill", d => getNodeColor(d, mode));
        }});

        function getNodeColor(d, mode) {{
            if (d.is_cycle) return '#FF6347';
            if (d.is_orphan) return '#FFFFE0';
            if (d.is_entry) return '#90EE90';

            let color = 'skyblue';

            if (mode === 'complexity') {{
                const complexity = d.complexity || 0;
                if (complexity > 50) color = '#FF4500';
                else if (complexity > 20) color = '#FFA500';
                else if (complexity > 5) color = '#FFD700';
            }} else if (mode === 'modified') {{
                 const hash = d.name.split('').reduce((acc, char) => char.charCodeAt(0) + ((acc << 5) - acc), 0);
                 color = d3.interpolateBlues(Math.abs(hash % 100) / 100);
            }} else if (mode === 'size') {{
                const size = d.size || 0;
                if (size > 100000) color = '#DC143C';
                else if (size > 20000) color = '#FF69B4';
                else if (size > 5000) color = '#DB7093';
            }}
            return color;
        }}

        window.addEventListener("resize", () => {{
            width = container.node().getBoundingClientRect().width;
            height = container.node().getBoundingClientRect().height;
            svg.attr("width", width).attr("height", height);
        }});

        </script>
    </body>
    </html>
    """
    return html_content

def _detect_cycles(graph: nx.DiGraph) -> List[List[str]]:
    try:
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            logger.warning(f"Detected {len(cycles)} circular dependencies.")
            for i, cycle in enumerate(cycles[:3]):
                 logger.warning(f"  Cycle {i+1}: {' -> '.join(cycle + [cycle[0]])}")
            if len(cycles) > 3:
                logger.warning("  (Additional cycles exist...)")
        else:
            logger.info("No circular dependencies detected.")
        return cycles
    except Exception as e:
        logger.error(f"Error during cycle detection: {e}")
        return []

def _calculate_module_metrics(file_path: str) -> Dict[str, int]:
    metrics = {'loc': 0, 'functions': 0, 'classes': 0}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        metrics['loc'] = len(lines)
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("def "):
                metrics['functions'] += 1
            elif stripped_line.startswith("class "):
                metrics['classes'] += 1
    except IOError as e:
        logger.warning(f"Could not read file for metrics calculation {file_path}: {e}")
    except Exception as e:
         logger.warning(f"Error processing file for metrics {file_path}: {e}")
    return metrics

def _get_orphan_modules(graph: nx.DiGraph, all_project_modules: Set[str]) -> Set[str]:
    orphans = set()
    if not graph.nodes():
        return all_project_modules

    nodes_in_graph = set(graph.nodes())
    importers = {target for _, target in graph.in_edges()}

    for module in all_project_modules:
        if module not in importers:
             is_self_importer_only = graph.has_edge(module, module) and graph.in_degree(module) == 1
             if not is_self_importer_only:
                 orphans.add(module)

    orphans.update(all_project_modules - nodes_in_graph)

    logger.info(f"Identified {len(orphans)} potential orphan modules.")
    return orphans

def _get_entry_point_modules(graph: nx.DiGraph, all_project_modules: Set[str]) -> Set[str]:
    entries = set()
    if not graph.nodes():
        return entries

    nodes_in_graph = set(graph.nodes())
    importers = {target for _, target in graph.in_edges()}
    importees = {source for source, _ in graph.out_edges()}

    for module in all_project_modules:
        is_imported = module in importers
        imports_others = module in importees or graph.out_degree(module) > 0

        if not is_imported:
            if imports_others or (module in all_project_modules and module not in nodes_in_graph):
                 is_self_importer_only = graph.has_edge(module, module) and graph.in_degree(module) == 1 and graph.out_degree(module) == 1
                 if not is_self_importer_only:
                    entries.add(module)

    logger.info(f"Identified {len(entries)} potential entry point modules.")
    return entries

def _update_import_counts(import_details: Dict[Tuple[str, str], Dict[str, Any]], from_module: str, to_module: str):
    edge = (from_module, to_module)
    if edge not in import_details:
        import_details[edge] = {"type": "direct", "count": 0}
    import_details[edge]["count"] += 1

class ModuleVisualizer:

    def __init__(self,
                 root_dir: str,
                 output_dir: str = None,
                 exclude_patterns: List[str] = None,
                 subpath: Optional[str] = None,
                 ignore_modules: Optional[List[str]] = None,
                 color_by: str = 'default'
                 ):
        self.root_dir = os.path.abspath(root_dir)
        self.output_dir = output_dir or os.path.join(self.root_dir, "module_viz")
        default_excludes = ["__pycache__", "*.pyc", "*.pyo", "*.pyd", ".venv", "venv", "env", ".env", "node_modules", ".git", ".hg", "*.egg-info"]
        self.exclude_patterns = default_excludes + (exclude_patterns or [])
        self.subpath_dir = os.path.abspath(os.path.join(self.root_dir, subpath)) if subpath else self.root_dir
        if not os.path.isdir(self.subpath_dir):
             logger.warning(f"Subpath '{subpath}' is not a valid directory within '{self.root_dir}'. Analyzing full root directory.")
             self.subpath_dir = self.root_dir
        elif not self.subpath_dir.startswith(self.root_dir):
             logger.warning(f"Subpath '{subpath}' is outside the root directory '{self.root_dir}'. Analyzing full root directory.")
             self.subpath_dir = self.root_dir
        else:
             logger.info(f"Analysis limited to subpath: {self.subpath_dir}")

        self.ignore_modules = set(ignore_modules or [])
        self.color_by = color_by

        self.modules: Dict[str, Dict[str, Any]] = {}
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.import_details: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(lambda: {"type": "direct", "count": 0})

        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")

        self.graph = nx.DiGraph()

        self.cycles: List[List[str]] = []
        self.orphan_modules: Set[str] = set()
        self.entry_point_modules: Set[str] = set()

    def scan_modules(self):
        logger.info(f"Scanning for Python modules in {self.subpath_dir}")

        if self.root_dir not in sys.path:
             logger.debug(f"Adding project root directory to sys.path: {self.root_dir}")
             sys.path.insert(0, self.root_dir)

        python_files = _find_python_files(self.subpath_dir, self.root_dir, self.exclude_patterns)

        for file_path in python_files:
            module_name = _get_module_name_from_path(file_path, self.root_dir)

            if not module_name:
                 logger.warning(f"Could not determine module name for file: {file_path}. Skipping.")
                 continue

            if module_name in self.ignore_modules:
                logger.debug(f"Ignoring module '{module_name}' as per ignore list.")
                continue

            metrics = _calculate_module_metrics(file_path)
            complexity_score = metrics['loc'] + metrics['functions'] * 5 + metrics['classes'] * 10

            self.modules[module_name] = {
                "path": file_path,
                "name": module_name,
                "size": os.path.getsize(file_path),
                "modified": os.path.getmtime(file_path),
                "loc": metrics['loc'],
                "functions": metrics['functions'],
                "classes": metrics['classes'],
                "complexity": complexity_score
            }
            logger.debug(f"Found module: {module_name} at {file_path}")

        logger.info(f"Found {len(self.modules)} Python modules within the target path.")
        if not self.modules:
            logger.warning("No Python modules found. Check the root directory, subpath, and exclude patterns.")


    def analyze_dependencies(self):
        if not self.modules:
            logger.warning("No modules found to analyze dependencies.")
            return

        logger.info("Analyzing module dependencies")
        total_imports_found = 0

        for module_name, module_info in self.modules.items():
            file_path = module_info["path"]
            imports = self._extract_imports(file_path)
            total_imports_found += len(imports)

            current_module_deps = set()
            for imp_module_base, _ in imports:
                if imp_module_base in self.ignore_modules:
                    logger.debug(f"Ignoring import of '{imp_module_base}' in '{module_name}' as per ignore list.")
                    continue

                if imp_module_base in self.modules:
                    if imp_module_base != module_name:
                        current_module_deps.add(imp_module_base)
                        _update_import_counts(self.import_details, module_name, imp_module_base)
                    else:
                        logger.debug(f"Ignoring self-import in module: {module_name}")
                else:
                    logger.debug(f"Module '{module_name}' imports external or unknown module: '{imp_module_base}'")

            self.dependencies[module_name] = current_module_deps

        logger.info("Building dependency graph")
        for module_name in self.modules:
            self.graph.add_node(module_name, **self.modules[module_name])

        edge_count = 0
        for module_name, deps in self.dependencies.items():
            if module_name not in self.graph: continue
            for dep in deps:
                if dep in self.graph:
                    edge_data = self.import_details.get((module_name, dep), {"type": "direct", "count": 1})
                    self.graph.add_edge(module_name, dep, **edge_data)
                    edge_count += 1
                else:
                     logger.debug(f"Dependency '{dep}' for module '{module_name}' not found in scanned modules. Skipping edge.")


        logger.info(f"Analyzed {total_imports_found} import statements.")
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

        self.cycles = _detect_cycles(self.graph)
        self.orphan_modules = _get_orphan_modules(self.graph, set(self.modules.keys()))
        self.entry_point_modules = _get_entry_point_modules(self.graph, set(self.modules.keys()))


    def _extract_imports(self, file_path: str) -> List[Tuple[str, Optional[str]]]:
        imports: List[Tuple[str, Optional[str]]] = []
        try:
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
            content = None
            for enc in encodings_to_try:
                 try:
                     with open(file_path, 'r', encoding=enc) as f:
                         content = f.read()
                     logger.debug(f"Successfully read {file_path} with encoding {enc}")
                     break
                 except UnicodeDecodeError:
                     logger.debug(f"Failed to read {file_path} with encoding {enc}")
                 except Exception as e_inner:
                     logger.warning(f"Error reading file {file_path}: {e_inner}")
                     return imports

            if content is None:
                 logger.warning(f"Could not decode file {file_path} with any attempted encoding. Skipping import extraction.")
                 return imports

            lines = content.splitlines()
            for line_num, line in enumerate(lines):
                try:
                    parsed_imports = _parse_import_line(line)
                    imports.extend(parsed_imports)
                except Exception as e_parse:
                    logger.warning(f"Error parsing imports on line {line_num+1} in {file_path}: {e_parse} | Line: '{line.strip()}'")

        except Exception as e:
            logger.error(f"Critical error extracting imports from {file_path}: {e}")

        logger.debug(f"Extracted {len(imports)} potential import references from {os.path.basename(file_path)}")
        return imports

    def generate_visualization(self):
        if not self.graph or self.graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, skipping static visualization generation.")
            return None

        logger.info("Generating static module visualization (PNG)")

        start_layout_time = time.time()
        try:
            pos = nx.kamada_kawai_layout(self.graph)
            logger.debug(f"Layout calculation took {time.time() - start_layout_time:.2f} seconds.")
        except Exception as e:
            logger.warning(f"Graph layout calculation failed: {e}. Falling back to random layout.")
            pos = nx.random_layout(self.graph, seed=42)


        node_sizes = [_calculate_node_size(data, min_size=300, max_size=4000, scale_factor=50)
                     for _, data in self.graph.nodes(data=True)]

        node_colors = [_calculate_node_color(data, self.color_by, node in self.orphan_modules, node in self.entry_point_modules, any(node in cycle for cycle in self.cycles))
                      for node, data in self.graph.nodes(data=True)]

        plt.figure(figsize=(max(12, self.graph.number_of_nodes() / 4), max(10, self.graph.number_of_nodes() / 6)))

        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.9
        )

        edge_widths = [1 + np.log1p(data.get('count', 1))/2 for u, v, data in self.graph.edges(data=True)]
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='gray',
            alpha=0.6,
            width=edge_widths,
            arrows=True,
            arrowsize=12,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.1'
        )

        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=max(6, 10 - self.graph.number_of_nodes() // 20),
            font_family='sans-serif',
            font_weight='normal'
        )

        plt.title(f"Module Dependencies ({os.path.basename(self.root_dir)})", fontsize=16)
        plt.axis('off')

        viz_path = os.path.join(self.output_dir, "module_dependencies.png")
        try:
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            logger.info(f"Module visualization saved to {viz_path}")
        except Exception as e:
            logger.error(f"Failed to save PNG visualization to {viz_path}: {e}")
            viz_path = None
        finally:
             plt.close()

        return viz_path

    def _generate_html_report(self):
        if not self.graph or self.graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, skipping HTML report generation.")
            return None

        logger.info("Generating interactive HTML report")
        report_path = os.path.join(self.output_dir, "module_report.html")

        graph_data = _generate_d3_data(self.graph, self.orphan_modules, self.entry_point_modules, self.cycles)

        stats = {
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
            'orphan_count': len(self.orphan_modules),
            'entry_count': len(self.entry_point_modules),
            'cycle_count': len(self.cycles)
        }

        html_content = _generate_html_template(graph_data, stats, self.cycles)

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Interactive HTML report saved to {report_path}")
            return report_path
        except IOError as e:
            logger.error(f"Failed to write HTML report to {report_path}: {e}")
            return None

    def export_graph_data(self, format: str = 'json') -> Optional[str]:
        if not self.graph:
            logger.warning("No graph data to export.")
            return None

        logger.info(f"Exporting graph data in {format} format")
        filename = f"module_graph.{format}"
        export_path = os.path.join(self.output_dir, filename)

        try:
            if format == 'json':
                data = nx.node_link_data(self.graph)
                if _save_json_data(data, export_path):
                    logger.info(f"Graph data saved to {export_path}")
                    return export_path
                else:
                     return None
            elif format == 'graphml':
                 g_export = self.graph.copy()
                 for node, data in g_export.nodes(data=True):
                     for key, value in data.items():
                         if not isinstance(value, (str, int, float, bool)):
                             g_export.nodes[node][key] = str(value)
                 nx.write_graphml(g_export, export_path, encoding='utf-8')
                 logger.info(f"Graph data saved to {export_path}")
                 return export_path
            else:
                logger.error(f"Unsupported export format: {format}. Use 'json' or 'graphml'.")
                return None
        except Exception as e:
            logger.error(f"Failed to export graph data to {export_path}: {e}")
            return None

    def compare_with_previous_snapshot(self, snapshot_file: str = "module_snapshot.json") -> Optional[Dict[str, Any]]:
        snapshot_path = os.path.join(self.output_dir, snapshot_file)
        logger.info(f"Attempting to compare with previous snapshot: {snapshot_path}")

        previous_data = _load_json_data(snapshot_path)

        if previous_data is None:
            logger.info("No previous snapshot found for comparison.")
            return None

        prev_nodes = len(previous_data.get('nodes', []))
        prev_links = len(previous_data.get('links', []))
        current_nodes = self.graph.number_of_nodes()
        current_edges = self.graph.number_of_edges()

        comparison = {
            "previous_nodes": prev_nodes,
            "current_nodes": current_nodes,
            "node_diff": current_nodes - prev_nodes,
            "previous_edges": prev_links,
            "current_edges": current_edges,
            "edge_diff": current_edges - prev_links,
            "snapshot_file": snapshot_path
        }

        logger.info(f"Comparison Results: Nodes changed by {comparison['node_diff']}, Edges changed by {comparison['edge_diff']}.")

        return comparison

    def save_snapshot(self, snapshot_file: str = "module_snapshot.json"):
        if not self.graph:
            logger.warning("No graph data to save as snapshot.")
            return

        snapshot_path = os.path.join(self.output_dir, snapshot_file)
        logger.info(f"Saving current graph state to snapshot: {snapshot_path}")
        data = nx.node_link_data(self.graph)
        _save_json_data(data, snapshot_path)


    def export_connections_table(self) -> Optional[str]:
        if not self.graph:
             logger.warning("Graph is empty, skipping connections table export.")
             return None

        logger.info("Exporting module connections as Markdown table")
        table_path = os.path.join(self.output_dir, "module_connections.md")

        connections = []
        for source, target, data in self.graph.edges(data=True):
            source_info = self.modules.get(source, {})
            target_info = self.modules.get(target, {})
            connections.append({
                "source": source,
                "source_path": os.path.relpath(source_info.get("path", ""), self.root_dir) if source_info.get("path") else "N/A",
                "target": target,
                "target_path": os.path.relpath(target_info.get("path", ""), self.root_dir) if target_info.get("path") else "N/A",
                "type": data.get("type", "direct"),
                "count": data.get("count", 1)
            })

        connections.sort(key=lambda x: (x["source"], x["target"]))

        md_content = f"""# Module Connections Report

- **Project Root**: `{self.root_dir}`
- **Analysis Path**: `{self.subpath_dir}`
- **Total Modules Scanned**: {len(self.modules)}
- **Total Dependencies Found**: {len(connections)}
- **Orphan Modules**: {len(self.orphan_modules)}
- **Entry Points**: {len(self.entry_point_modules)}
- **Circular Dependencies**: {len(self.cycles)}
- **Generated**: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Connection Table

| Source Module | Target Module | Import Count | Relative Source Path |
|---------------|---------------|--------------|----------------------|
"""
        for conn in connections:
            md_content += f"| `{conn['source']}` | `{conn['target']}` | {conn['count']} | `{conn['source_path']}` |\n"

        md_content += "\n## Special Modules\n\n"
        if self.orphan_modules:
            md_content += "### Orphan Modules (Not imported by other scanned modules)\n"
            for module in sorted(list(self.orphan_modules)):
                md_content += f"- `{module}`\n"
            md_content += "\n"
        else:
            md_content += "No orphan modules identified.\n\n"

        if self.entry_point_modules:
             md_content += "### Potential Entry Points (Import others, but not imported themselves)\n"
             for module in sorted(list(self.entry_point_modules)):
                 md_content += f"- `{module}`\n"
             md_content += "\n"
        else:
             md_content += "No potential entry point modules identified.\n\n"

        if self.cycles:
            md_content += "### Circular Dependencies\n"
            for i, cycle in enumerate(self.cycles):
                md_content += f"- Cycle {i+1}: `{'` -> `'.join(cycle + [cycle[0]])}`\n"
            md_content += "\n"
        else:
            md_content += "No circular dependencies detected.\n\n"


        try:
            with open(table_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            logger.info(f"Module connections table saved to {table_path}")
            return table_path
        except IOError as e:
            logger.error(f"Failed to write connections table to {table_path}: {e}")
            return None


    def run(self, compare: bool = False, export_format: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"Starting module visualization for: {self.root_dir}")

        self.scan_modules()
        self.analyze_dependencies()

        comparison_results = None
        if compare and self.graph:
            comparison_results = self.compare_with_previous_snapshot()

        png_path = self.generate_visualization()
        html_path = self._generate_html_report()
        table_path = self.export_connections_table()

        export_path = None
        if export_format and self.graph:
            export_path = self.export_graph_data(format=export_format)

        if self.graph:
            self.save_snapshot()

        duration = time.time() - start_time
        logger.info(f"Visualization process completed in {duration:.2f} seconds")

        return {
            "modules_count": len(self.modules),
            "dependencies_count": self.graph.number_of_edges() if self.graph else 0,
            "output_dir": self.output_dir,
            "png_path": png_path,
            "html_path": html_path,
            "table_path": table_path,
            "export_path": export_path,
            "cycles_found": len(self.cycles),
            "orphans_found": len(self.orphan_modules),
            "entry_points_found": len(self.entry_point_modules),
            "comparison": comparison_results,
            "duration_seconds": duration
        }

def main():
    parser = argparse.ArgumentParser(description="Generate Python module dependency visualizations.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root_dir", default=".", nargs='?', help="Root directory of the Python project to analyze.")
    parser.add_argument("--output-dir", "-o", help="Directory for output files (default: <root_dir>/module_viz).")
    parser.add_argument("--exclude", action="append", help="Patterns to exclude (e.g., 'tests', '*.temp.py'). Can be used multiple times.")
    parser.add_argument("--table-only", action="store_true", help="Generate only the Markdown connections table.")
    parser.add_argument("--html-only", action="store_true", help="Generate only the interactive HTML report.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (DEBUG level).")

    parser.add_argument("--subpath", "-s", help="Analyze only a specific sub-directory within the root directory.")
    parser.add_argument("--ignore-module", dest="ignore_modules", action="append", help="Specific module names to ignore during analysis (e.g., 'config'). Can be used multiple times.")
    parser.add_argument("--color-by", choices=['default', 'complexity', 'modified', 'size'], default='default', help="Color nodes in visualizations based on the selected metric.")
    parser.add_argument("--export-graph", choices=['json', 'graphml'], help="Export the raw graph data to the specified format.")
    parser.add_argument("--compare", action="store_true", help="Compare current analysis with the last saved snapshot (module_snapshot.json).")
    parser.add_argument("--detect-cycles", action="store_true", help="Explicitly run and report circular dependencies (runs by default, this flag ensures reporting focus).")


    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled.")

    try:
        root_dir = os.path.abspath(args.root_dir)
        if not os.path.isdir(root_dir):
             parser.error(f"Root directory not found or is not a directory: {args.root_dir}")
    except Exception as e:
         parser.error(f"Error processing root directory path '{args.root_dir}': {e}")


    visualizer = ModuleVisualizer(
        root_dir=root_dir,
        output_dir=args.output_dir,
        exclude_patterns=args.exclude,
        subpath=args.subpath,
        ignore_modules=args.ignore_modules,
        color_by=args.color_by
    )

    result = None
    if args.table_only:
        logger.info("Running in table-only mode.")
        visualizer.scan_modules()
        visualizer.analyze_dependencies()
        table_path = visualizer.export_connections_table()
        if table_path:
            print(f"\nConnections table saved to: {table_path}")
        else:
            print("\nFailed to generate connections table.")
            return 1
    elif args.html_only:
        logger.info("Running in html-only mode.")
        visualizer.scan_modules()
        visualizer.analyze_dependencies()
        html_path = visualizer._generate_html_report()
        if html_path:
             print(f"\nHTML report saved to: {html_path}")
        else:
            print("\nFailed to generate HTML report.")
            return 1
    else:
        result = visualizer.run(compare=args.compare, export_format=args.export_graph)

        if result:
            print("\n--- Visualization Summary ---")
            print(f"- Project Root: {visualizer.root_dir}")
            if visualizer.subpath_dir != visualizer.root_dir:
                 print(f"- Analyzed Subpath: {visualizer.subpath_dir}")
            print(f"- Modules analyzed: {result['modules_count']}")
            print(f"- Dependencies found: {result['dependencies_count']}")
            print(f"- Cycles detected: {result['cycles_found']}")
            print(f"- Orphan modules: {result['orphans_found']}")
            print(f"- Entry points: {result['entry_points_found']}")
            if result['png_path']: print(f"- PNG graph saved to: {result['png_path']}")
            if result['html_path']: print(f"- Interactive HTML report saved to: {result['html_path']}")
            if result['table_path']: print(f"- Connections table saved to: {result['table_path']}")
            if result['export_path']: print(f"- Graph data export saved to: {result['export_path']}")
            if result['comparison']:
                 print("- Comparison with previous snapshot:")
                 comp = result['comparison']
                 print(f"  - Snapshot file: {os.path.basename(comp['snapshot_file'])}")
                 print(f"  - Node change: {comp['node_diff']:+d} ({comp['previous_nodes']} -> {comp['current_nodes']})")
                 print(f"  - Edge change: {comp['edge_diff']:+d} ({comp['previous_edges']} -> {comp['current_edges']})")
            print(f"- Analysis duration: {result['duration_seconds']:.2f} seconds")
            print(f"- Output directory: {result['output_dir']}")
            print("-----------------------------")
        else:
            print("\nVisualization process failed to produce results.")
            return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())