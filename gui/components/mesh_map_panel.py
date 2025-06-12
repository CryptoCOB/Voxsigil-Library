from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QScrollArea, QPushButton, QFrame, QSplitter,
                            QTreeWidget, QTreeWidgetItem, QTextEdit)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QFont
import json

class MeshMapPanel(QWidget):
    """Enhanced panel displaying the current agent mesh graph with visualization."""
    
    agent_selected = pyqtSignal(str)  # Signal when agent is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph_data = {}
        self.setup_ui()
        
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
            agent_connections = [conn for conn in connections 
                               if conn.get("from") == agent_name or conn.get("to") == agent_name]
            
            if agent_connections:
                conn_item = QTreeWidgetItem([f"Connections ({len(agent_connections)})"])
                item.addChild(conn_item)
                
                for conn in agent_connections:
                    other_agent = conn.get("to") if conn.get("from") == agent_name else conn.get("from")
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
        agent_connections = [conn for conn in connections 
                           if conn.get("from") == name or conn.get("to") == name]
        
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
        """Clear all graph data."""
        self.refresh({})


class GraphVisualizationWidget(QWidget):
    """Widget for visualizing the mesh network graph."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph_data = {}
        self.setMinimumSize(300, 200)
        
    def update_graph(self, graph: dict):
        """Update the graph visualization."""
        self.graph_data = graph
        self.update()
        
    def clear(self):
        """Clear the visualization."""
        self.graph_data = {}
        self.update()
        
    def paintEvent(self, event):
        """Paint the graph visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if not self.graph_data:
            painter.setPen(QPen(QColor(128, 128, 128), 1))
            painter.drawText(self.rect(), Qt.AlignCenter, "No graph data to display")
            return
            
        agents = self.graph_data.get("agents", {})
        connections = self.graph_data.get("connections", [])
        
        if not agents:
            painter.setPen(QPen(QColor(128, 128, 128), 1))
            painter.drawText(self.rect(), Qt.AlignCenter, "No agents in graph")
            return
            
        # Calculate positions for agents in a circle
        import math
        center_x = self.width() // 2
        center_y = self.height() // 2
        radius = min(center_x, center_y) - 50
        
        agent_positions = {}
        agent_list = list(agents.keys())
        
        for i, agent_name in enumerate(agent_list):
            angle = 2 * math.pi * i / len(agent_list)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            agent_positions[agent_name] = (x, y)
            
        # Draw connections
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        for conn in connections:
            from_agent = conn.get("from")
            to_agent = conn.get("to")
            
            if from_agent in agent_positions and to_agent in agent_positions:
                from_pos = agent_positions[from_agent]
                to_pos = agent_positions[to_agent]
                painter.drawLine(int(from_pos[0]), int(from_pos[1]), 
                               int(to_pos[0]), int(to_pos[1]))
                
        # Draw agents
        for agent_name, agent_data in agents.items():
            if agent_name in agent_positions:
                x, y = agent_positions[agent_name]
                
                # Get color
                if isinstance(agent_data, dict) and "color" in agent_data:
                    color = QColor(agent_data["color"])
                elif isinstance(agent_data, str):
                    color = QColor(agent_data)
                else:
                    color = QColor(128, 128, 128)
                    
                # Draw agent circle
                painter.setBrush(color)
                painter.setPen(QPen(QColor(0, 0, 0), 2))
                painter.drawEllipse(int(x-15), int(y-15), 30, 30)
                
                # Draw agent name
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawText(int(x-30), int(y+35), 60, 20, Qt.AlignCenter, agent_name)
