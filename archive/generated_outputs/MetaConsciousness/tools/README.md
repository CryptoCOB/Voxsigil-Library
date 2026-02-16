# MetaConsciousness Development Tools

This directory contains utility tools for development, debugging, and analysis of the MetaConsciousness system.

## Available Tools

### Module Visualization Tool

The `visualize_modules.py` script generates comprehensive visualizations and analysis of project module dependencies.

**Features:**
- Advanced dependency analysis with cycle detection
- Multiple visualization formats:
  - Static PNG visualization with customizable layouts
  - Interactive HTML visualization with D3.js
  - Dependency matrix visualization
  - Markdown connection tables
- Connectivity metrics:
  - Module coupling analysis
  - Orphan module detection
  - Entry point identification
  - Circular dependency detection
- Visualization options:
  - Color nodes by complexity, modification time, or size
  - Adjustable layout algorithms
  - Customizable node and edge styling
- Advanced filtering:
  - Module pattern exclusion
  - Subpath analysis support
  - Module ignore lists
- Analysis features:
  - Snapshot comparison
  - Graph data export (JSON/GraphML)
  - Detailed metrics reporting

**Usage:**
```bash
# Basic visualization
python tools/visualize_modules.py

# Analyze specific subdirectory
python tools/visualize_modules.py --subpath=MetaConsciousness/core

# Color nodes by complexity
python tools/visualize_modules.py --color-by=complexity

# Compare with previous snapshot
python tools/visualize_modules.py --compare

# Export graph data
python tools/visualize_modules.py --export-graph=json

# Analyze with exclusions
python tools/visualize_modules.py --exclude="tests/*" --ignore-module=config
```

**Output Files:**
- `module_dependencies.png` - Static visualization
- `connectivity_matrix.png` - Dependency matrix visualization
- `module_report.html` - Interactive D3.js visualization
- `module_connections.md` - Detailed connection analysis
- `module_snapshot.json` - Snapshot for comparisons
- `module_graph.json/graphml` - Raw graph data (optional)

**Connection Analysis:**
The `module_connections.md` report includes:
- Project statistics and metrics
- Connection tables with import counts
- Special module identification:
  - Orphaned modules
  - Entry points
  - Circular dependencies
- Module dependency hierarchies
- Complexity and coupling metrics

**Visualization Features:**
- Interactive graph navigation
- Node highlighting and filtering
- Dependency path tracing
- Zoom and pan controls
- Search functionality
- Customizable node coloring
- Tooltip details

## Running the Tools

Run directly from the project root:

```bash
python tools/visualize_modules.py [options]
```

## Command Line Options

```
--root-dir DIR         Root directory to analyze
--subpath DIR         Analyze specific subdirectory
--output-dir DIR      Custom output location
--exclude PATTERN     Exclude patterns (multiple allowed)
--ignore-module MOD   Ignore specific modules
--color-by MODE       Node coloring (complexity/modified/size)
--export-graph FMT    Export graph data (json/graphml)
--compare            Compare with previous snapshot
--table-only         Generate only the connections table
--html-only          Generate only the HTML report
--detect-cycles      Focus on circular dependency detection
--verbose            Enable detailed logging
```

## Adding New Tools

When adding new development tools:
1. Place them in this `tools` directory
2. Add documentation to this README
3. Ensure they can be run from the project root
