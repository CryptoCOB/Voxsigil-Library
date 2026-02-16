# Module Dependency Visualizer

A comprehensive Python module dependency visualization and analysis tool that helps you understand and improve your project's architecture.

## Features

- **Dependency Visualization**: Generate interactive and static visualizations of module dependencies
- **Cycle Detection**: Identify circular dependencies that can lead to design problems
- **Metrics Calculation**: Analyze complexity, coupling, and cohesion metrics
- **Architecture Analysis**: Detect common architectural patterns in your codebase
- **API Surface Analysis**: Understand the public interfaces exposed by your modules
- **Change Impact Analysis**: Predict the impact of changes to specific modules
- **Technical Debt Estimation**: Identify areas that need refactoring attention
- **Dependency Validation**: Verify dependencies and identify potential issues
- **Migration Planning**: Plan the optimal order for updating dependencies
- **Real-time Monitoring**: Watch for changes and update visualizations automatically

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# Basic usage
python -m tools.visualizer C:\Users\16479\Desktop\MetaConsciouness

# Analyze a specific subpath
python -m tools.visualizer <project-root-dir> --subpath <subpath>

# Exclude specific patterns
python -m tools.visualizer <project-root-dir> --exclude tests --exclude vendor

# Generate specific output formats
python -m tools.visualizer <project-root-dir> --format json

# Compare with previous analysis
python -m tools.visualizer <project-root-dir> --compare

# Run in interactive mode
python -m tools.visualizer <project-root-dir> --interactive
```

### Interactive Mode

The tool provides an interactive shell mode that allows you to:

- Focus on specific modules and their dependencies
- Display detailed metrics
- Analyze dependency cycles
- Validate project dependencies
- Analyze change impact
- Assess module health scores

### Python API

```python
from tools.visualizer import ModuleVisualizer

# Initialize the visualizer
visualizer = ModuleVisualizer(
    root_dir='path/to/project',
    exclude_patterns=['tests', 'docs'],
    color_by='complexity'
)

# Run analysis
result = visualizer.run()

# Focus on a specific module
module_result = visualizer.focus_module('module.name')

# Validate dependencies
validation = visualizer.validate_project()

# Get detailed metrics
metrics = visualizer.show_metrics()
```

## Output Files

The tool generates the following outputs:

- `module_report.html`: Interactive visualization with D3.js
- `module_graph.png`: Static visualization of dependencies
- `module_dependencies.md`: Markdown table of dependencies
- `module_snapshot.json`: JSON representation of the current state (for comparisons)
- `project_documentation.md`: Generated documentation based on analysis

## Advanced Features

### Architecture Analysis

The tool can detect common architectural patterns:

- Layered Architecture
- Facade Pattern
- Mediator Pattern
- Singleton Pattern

### Dependency Migration Planning

For complex refactoring or upgrades, the tool can generate a dependency migration plan:

```python
from tools.visualizer.planners import MigrationPlanner

planner = MigrationPlanner()
plan = planner.plan_dependency_migration(graph, {'module1', 'module2'})
```

### Real-time Monitoring

Monitor project changes in real-time:

```python
from tools.visualizer.monitors import DependencyMonitor

def on_change():
    visualizer.run(compare=True)

monitor = DependencyMonitor(on_change)
# Start monitoring
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
