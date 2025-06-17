import logging
import random

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Import the visualization widget used for displaying the mesh graph
# from .graph_visualization_widget import GraphVisualizationWidget  # Temporarily commented out


# Placeholder widget until graph_visualization_widget is implemented
class GraphVisualizationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        label = QLabel("Graph Visualization Widget\n(Implementation pending)")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        self.setLayout(layout)

    def update_graph(self, graph_data):
        """Placeholder method for graph updates"""
        pass


logger = logging.getLogger(__name__)


class MeshMapPanel(QWidget):
    """Enhanced panel displaying the current agent mesh graph with visualization."""

    agent_selected = pyqtSignal(str)  # Signal when agent is selected

    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph_data = {}
        self.setup_ui()

        # Generate initial mesh data
        self.generate_initial_mesh_data()

        # Setup auto-refresh timer for holo mesh (after everything is initialized)
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_refresh_mesh)
        self.refresh_timer.start(5000)  # Refresh every 5 seconds

    def setup_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Agent Mesh Network")
        header.setFont(QFont("Arial", 12, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left side - Agent list and controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right side - Graph visualization
        self.graph_widget = GraphVisualizationWidget()
        splitter.addWidget(self.graph_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        # Status bar
        self.status_label = QLabel("No graph data loaded")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)

    def create_left_panel(self):
        """Create the left control panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Agent tree view
        self.agent_tree = QTreeWidget()
        self.agent_tree.setHeaderLabel("Agents")
        self.agent_tree.itemClicked.connect(self.on_agent_selected)
        layout.addWidget(self.agent_tree)

        # Controls
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.manual_refresh)
        controls_layout.addWidget(self.refresh_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_graph)
        controls_layout.addWidget(self.clear_btn)

        layout.addWidget(controls_frame)
        # Details text area
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(100)
        self.details_text.setPlaceholderText("Select an agent to view details...")
        layout.addWidget(self.details_text)

        return widget

    def refresh(self, graph: dict) -> None:
        """Refresh panel contents with new graph data."""
        self.graph_data = graph

        if not graph:
            self.status_label.setText("No graph data available")
            self.agent_tree.clear()
            self.graph_widget.clear()
            self.details_text.clear()
            return

        self.update_agent_tree()
        self.graph_widget.update_graph(graph)

        agent_count = len(graph.get("agents", {}))
        connection_count = len(graph.get("connections", []))
        self.status_label.setText(f"Agents: {agent_count} | Connections: {connection_count}")

    def update_graph(self, graph: dict) -> None:
        """Alias for refresh method to maintain compatibility."""
        self.refresh(graph)

    def update_agent_tree(self):
        """Update the agent tree view."""
        self.agent_tree.clear()

        agents = self.graph_data.get("agents", {})
        connections = self.graph_data.get("connections", [])

        for agent_name, agent_data in agents.items():
            item = QTreeWidgetItem([agent_name])

            # Set color indicator
            if isinstance(agent_data, dict) and "color" in agent_data:
                color = agent_data["color"]
            elif isinstance(agent_data, str):
                color = agent_data
            else:
                color = "#808080"

            item.setData(0, Qt.UserRole, {"name": agent_name, "data": agent_data})
            item.setText(0, f"● {agent_name}")
            item.setForeground(0, QColor(color))

            # Add connection info as children
            agent_connections = [
                conn
                for conn in connections
                if conn.get("from") == agent_name or conn.get("to") == agent_name
            ]

            if agent_connections:
                conn_item = QTreeWidgetItem([f"Connections ({len(agent_connections)})"])
                item.addChild(conn_item)

                for conn in agent_connections:
                    other_agent = (
                        conn.get("to") if conn.get("from") == agent_name else conn.get("from")
                    )
                    conn_detail = QTreeWidgetItem([f"→ {other_agent}"])
                    conn_item.addChild(conn_detail)

            self.agent_tree.addTopLevelItem(item)

    def on_agent_selected(self, item, column):
        """Handle agent selection."""
        agent_data = item.data(0, Qt.UserRole)
        if agent_data:
            agent_name = agent_data["name"]
            self.show_agent_details(agent_name, agent_data["data"])
            self.agent_selected.emit(agent_name)

    def show_agent_details(self, name: str, data):
        """Show detailed information about selected agent."""
        details = f"Agent: {name}\n"

        if isinstance(data, dict):
            for key, value in data.items():
                details += f"{key}: {value}\n"
        else:
            details += f"Color: {data}\n"

        # Add connection info
        connections = self.graph_data.get("connections", [])
        agent_connections = [
            conn for conn in connections if conn.get("from") == name or conn.get("to") == name
        ]

        if agent_connections:
            details += f"\nConnections ({len(agent_connections)}):\n"
            for conn in agent_connections:
                other = conn.get("to") if conn.get("from") == name else conn.get("from")
                details += f"  → {other}\n"

        self.details_text.setText(details)

    def manual_refresh(self):
        """Manually refresh with current graph data."""
        self.refresh(self.graph_data)

    def clear_graph(self):
        """clear the current graph data and UI."""
        self.graph_data = {}
        self.agent_tree.clear()
        self.graph_widget.clear()
        self.status_label.setText("Graph cleared")
        self.details_text.clear()

    def generate_initial_mesh_data(self):
        """Generate initial mesh data for visualization"""
        try:
            import random

            # Define core components with display colours
            core_components = [
                ("VoxSigil Core", "#FF6F61"),
                ("AI Engine", "#6B5B95"),
                ("Memory Bank", "#88B04B"),
                ("Training System", "#F7CAC9"),
                ("Voice Interface", "#92A8D1"),
                ("Mesh Network", "#955251"),
                ("ARC Processor", "#B565A7"),
                ("Vanta Core", "#009B77"),
                ("Neural Bridge", "#DD4124"),
                ("Data Flow", "#45B8AC"),
            ]

            # Build agents dictionary
            agents = {
                name: {"color": colour, "status": "active"} for name, colour in core_components
            }

            # Randomly create connections between components
            connections = []
            names_only = [name for name, _ in core_components]
            for i, src in enumerate(names_only):
                for dst in names_only[i + 1 :]:
                    if random.random() < 0.3:  # 30 % probability
                        connections.append({"from": src, "to": dst})

            # Store and display the graph
            self.graph_data = {"agents": agents, "connections": connections}
            self.refresh(self.graph_data)

        except Exception as e:
            logger.error(f"Error generating initial mesh data: {e}")

        except Exception as e:
            logger.error(f"Error generating initial mesh data: {e}")

    def auto_refresh_mesh(self):
        """Auto-refresh mesh data with live updates"""
        try:
            import random

            # Simulate dynamic mesh activity
            if hasattr(self.graph_widget, "nodes") and self.graph_widget.nodes:
                # Update node statuses and loads
                for node in self.graph_widget.nodes:
                    # Randomly update status
                    if random.random() < 0.1:  # 10% chance to change status
                        node["status"] = random.choice(["active", "processing", "idle", "training"])

                    # Update load gradually
                    current_load = node.get("load", 50)
                    change = random.randint(-5, 5)
                    node["load"] = max(0, min(100, current_load + change))
                    # Update connection count if your implementation tracks it
                    # (Placeholder for future logic)

                self.update_agent_tree()
                if hasattr(self, "graph_widget"):
                    # Update the visual graph representation
                    self.graph_widget.update_graph(self.graph_data)

                    # Update edges if the widget tracks them
                    if hasattr(self.graph_widget, "edges") and self.graph_widget.edges:
                        for edge in self.graph_widget.edges:
                            if random.random() < 0.2:  # 20% chance to toggle activity
                                edge["active"] = not edge.get("active", True)

                            # Update strength
                            current_strength = edge.get("strength", 0.5)
                            change = random.uniform(-0.1, 0.1)
                            edge["strength"] = max(0.1, min(1.0, current_strength + change))

                    # Trigger visualization update
                    self.graph_widget.update()

        except Exception as e:
            logger.error(f"Error auto-refreshing mesh: {e}")

    def add_real_time_node(self, name, node_type, position, status="active"):
        """Add a new node to the mesh in real-time"""
        try:
            if hasattr(self.graph_widget, "nodes"):
                new_id = len(self.graph_widget.nodes)
                new_node = {
                    "id": new_id,
                    "name": name,
                    "type": node_type,
                    "x": position[0],
                    "y": position[1],
                    "status": status,
                    "load": random.randint(10, 60),
                    "connections": 1,
                }
                self.graph_widget.nodes.append(new_node)
                self.graph_widget.update()
                logger.info(f"Added real-time node: {name}")

        except Exception as e:
            logger.error(f"Error adding real-time node: {e}")

    def update_mesh_status(self, component_name, new_status, load=None):
        """Update the status of a specific mesh component"""
        try:
            if hasattr(self.graph_widget, "nodes"):
                for node in self.graph_widget.nodes:
                    if node["name"] == component_name:
                        node["status"] = new_status
                        if load is not None:
                            node["load"] = load
                        self.graph_widget.update()
                        break

        except Exception as e:
            logger.error(f"Error updating mesh status: {e}")

    # ...existing code...
