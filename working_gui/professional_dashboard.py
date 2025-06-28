#!/usr/bin/env python3
"""
VoxSigil Professional Dashboard - Ultimate Streamlined GUI
=========================================================
The cleanest, most professional interface for VoxSigil training
with complete VantaCore component visibility and management.
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "Vanta"))

# PyQt5 imports
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    print("✅ PyQt5 Professional Dashboard Ready")
except ImportError as e:
    print(f"❌ PyQt5 import failed: {e}")
    sys.exit(1)

# VantaCore imports
try:
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore
    print("✅ VantaCore Professional Integration Ready")
except ImportError as e:
    print(f"❌ VantaCore import failed: {e}")
    UnifiedVantaCore = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComponentCard(QFrame):
    """Professional card widget for individual components"""
    
    def __init__(self, name: str, component_type: str, status: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.component_type = component_type
        self.status = status
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the card UI"""
        self.setFrameStyle(QFrame.Box)
        self.setFixedSize(250, 100)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        
        # Header with icon and name
        header_layout = QHBoxLayout()
        
        # Type icon
        icon_label = QLabel(self.get_type_icon())
        icon_label.setFont(QFont("Arial", 16))
        header_layout.addWidget(icon_label)
        
        # Component name
        name_label = QLabel(self.get_display_name())
        name_label.setFont(QFont("Arial", 10, QFont.Bold))
        name_label.setWordWrap(True)
        header_layout.addWidget(name_label)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Status
        status_label = QLabel(f"Status: {self.get_status_display()}")
        status_label.setFont(QFont("Arial", 9))
        status_label.setStyleSheet(f"color: {self.get_status_color()};")
        layout.addWidget(status_label)
        
        # Type badge
        type_label = QLabel(self.component_type.title())
        type_label.setFont(QFont("Arial", 8))
        type_label.setStyleSheet("""
            background-color: #e9ecef;
            border-radius: 3px;
            padding: 2px 6px;
            color: #495057;
        """)
        type_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(type_label)
        
        # Apply card styling
        self.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                border-left: 4px solid {self.get_type_color()};
            }}
            QFrame:hover {{
                border: 1px solid #007bff;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
        """)
        
    def get_type_icon(self) -> str:
        """Get icon for component type"""
        icons = {
            'training': '🎯',
            'evaluation': '📊',
            'inference': '🧠',
            'visualization': '📈',
            'system': '⚙️'
        }
        return icons.get(self.component_type.lower(), '📦')
        
    def get_type_color(self) -> str:
        """Get color for component type"""
        colors = {
            'training': '#28a745',
            'evaluation': '#17a2b8',
            'inference': '#6f42c1',
            'visualization': '#fd7e14',
            'system': '#6c757d'
        }
        return colors.get(self.component_type.lower(), '#6c757d')
        
    def get_display_name(self) -> str:
        """Get clean display name"""
        name = self.name
        # Remove type prefix
        for prefix in ['training_', 'evaluation_', 'inference_', 'visualization_']:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name.replace('_', ' ').title()
        
    def get_status_display(self) -> str:
        """Get formatted status"""
        return "Active" if self.status == "✅" else "Inactive"
        
    def get_status_color(self) -> str:
        """Get status color"""
        return "#28a745" if self.status == "✅" else "#dc3545"


class ComponentsGalleryWidget(QScrollArea):
    """Professional gallery view of components"""
    
    def __init__(self, vanta_core=None):
        super().__init__()
        self.vanta_core = vanta_core
        self.setup_ui()
        self.refresh_components()
        
    def setup_ui(self):
        """Setup the gallery UI"""
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create scroll content widget
        content_widget = QWidget()
        self.setWidget(content_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(content_widget)
        self.main_layout.setSpacing(20)
        
    def refresh_components(self):
        """Refresh component display"""
        # Clear existing widgets
        for i in reversed(range(self.main_layout.count())):
            self.main_layout.itemAt(i).widget().setParent(None)
            
        if not self.vanta_core or not hasattr(self.vanta_core, 'registry'):
            no_data_label = QLabel("❌ VantaCore not available")
            no_data_label.setAlignment(Qt.AlignCenter)
            no_data_label.setFont(QFont("Arial", 14))
            self.main_layout.addWidget(no_data_label)
            return
            
        try:
            # Get components organized by type
            component_names = self.vanta_core.registry.list_components()
            
            components_by_type = {
                'Training': [],
                'Evaluation': [],
                'Inference': [],
                'Visualization': [],
                'System': []
            }
            
            for name in component_names:
                if name.startswith('training_'):
                    components_by_type['Training'].append(name)
                elif name.startswith('evaluation_'):
                    components_by_type['Evaluation'].append(name)
                elif name.startswith('inference_'):
                    components_by_type['Inference'].append(name)
                elif name.startswith('visualization_'):
                    components_by_type['Visualization'].append(name)
                else:
                    components_by_type['System'].append(name)
            
            # Create sections
            for category, components in components_by_type.items():
                if not components:
                    continue
                    
                # Section header
                header = QLabel(f"{self.get_category_icon(category)} {category} Components ({len(components)})")
                header.setFont(QFont("Arial", 14, QFont.Bold))
                header.setStyleSheet(f"""
                    color: {self.get_category_color(category)};
                    padding: 10px 0px 5px 0px;
                """)
                self.main_layout.addWidget(header)
                
                # Components grid
                grid_widget = QWidget()
                grid_layout = QGridLayout(grid_widget)
                grid_layout.setSpacing(10)
                
                row, col = 0, 0
                for comp_name in sorted(components):
                    try:
                        component = self.vanta_core.registry.get(comp_name)
                        status = "✅" if component else "❌"
                        
                        card = ComponentCard(comp_name, category.lower(), status)
                        grid_layout.addWidget(card, row, col)
                        
                        col += 1
                        if col >= 4:  # 4 cards per row
                            col = 0
                            row += 1
                            
                    except Exception as e:
                        logger.error(f"Error creating card for {comp_name}: {e}")
                
                self.main_layout.addWidget(grid_widget)
                
        except Exception as e:
            logger.error(f"Error refreshing components: {e}")
            error_label = QLabel(f"❌ Error loading components: {str(e)}")
            error_label.setAlignment(Qt.AlignCenter)
            self.main_layout.addWidget(error_label)
            
        self.main_layout.addStretch()
        
    def get_category_icon(self, category: str) -> str:
        """Get icon for category"""
        icons = {
            'Training': '🎯',
            'Evaluation': '📊', 
            'Inference': '🧠',
            'Visualization': '📈',
            'System': '⚙️'
        }
        return icons.get(category, '📦')
        
    def get_category_color(self, category: str) -> str:
        """Get color for category"""
        colors = {
            'Training': '#28a745',
            'Evaluation': '#17a2b8',
            'Inference': '#6f42c1',
            'Visualization': '#fd7e14',
            'System': '#6c757d'
        }
        return colors.get(category, '#6c757d')


class ProfessionalStatusPanel(QWidget):
    """Professional status panel with metrics"""
    
    def __init__(self, vanta_core=None):
        super().__init__()
        self.vanta_core = vanta_core
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        """Setup the status panel UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("📊 System Dashboard")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Metrics grid
        metrics_frame = QFrame()
        metrics_frame.setFrameStyle(QFrame.Box)
        metrics_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        
        metrics_layout = QGridLayout(metrics_frame)
        
        # Component count
        self.component_metric = self.create_metric_widget("📦", "Components", "Loading...")
        metrics_layout.addWidget(self.component_metric, 0, 0)
        
        # Agent count
        self.agent_metric = self.create_metric_widget("🤖", "Agents", "Loading...")
        metrics_layout.addWidget(self.agent_metric, 0, 1)
        
        # System status
        self.status_metric = self.create_metric_widget("⚡", "Status", "Loading...")
        metrics_layout.addWidget(self.status_metric, 1, 0)
        
        # Uptime
        self.uptime_metric = self.create_metric_widget("⏰", "Uptime", "Loading...")
        metrics_layout.addWidget(self.uptime_metric, 1, 1)
        
        layout.addWidget(metrics_frame)
        
        # Activity log
        log_label = QLabel("📝 Recent Activity")
        log_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(log_label)
        
        self.activity_log = QTextEdit()
        self.activity_log.setMaximumHeight(150)
        self.activity_log.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.activity_log)
        
        layout.addStretch()
        
        # Record startup
        self.start_time = datetime.now()
        self.log_activity("System initialized")
        
    def create_metric_widget(self, icon: str, label: str, value: str) -> QWidget:
        """Create a metric display widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(5)
        
        # Icon
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Arial", 20))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        # Value
        value_label = QLabel(value)
        value_label.setFont(QFont("Arial", 14, QFont.Bold))
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setObjectName("metric_value")
        layout.addWidget(value_label)
        
        # Label
        label_label = QLabel(label)
        label_label.setFont(QFont("Arial", 10))
        label_label.setAlignment(Qt.AlignCenter)
        label_label.setStyleSheet("color: #6c757d;")
        layout.addWidget(label_label)
        
        widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        
        return widget
        
    def setup_timer(self):
        """Setup update timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(3000)  # Update every 3 seconds
        self.update_metrics()  # Initial update
        
    def update_metrics(self):
        """Update all metrics"""
        try:
            if self.vanta_core and hasattr(self.vanta_core, 'registry'):
                # Component count
                components = self.vanta_core.registry.list_components()
                self.update_metric_value(self.component_metric, str(len(components)))
                
                # Agent count
                if hasattr(self.vanta_core, 'get_all_agents'):
                    agents = self.vanta_core.get_all_agents()
                    self.update_metric_value(self.agent_metric, str(len(agents)))
                
                # System status
                self.update_metric_value(self.status_metric, "Operational")
                
            else:
                self.update_metric_value(self.component_metric, "N/A")
                self.update_metric_value(self.agent_metric, "N/A")
                self.update_metric_value(self.status_metric, "Disconnected")
            
            # Uptime
            uptime = datetime.now() - self.start_time
            uptime_str = f"{uptime.seconds // 60}m {uptime.seconds % 60}s"
            self.update_metric_value(self.uptime_metric, uptime_str)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            
    def update_metric_value(self, widget: QWidget, value: str):
        """Update a metric widget's value"""
        value_label = widget.findChild(QLabel, "metric_value")
        if value_label:
            value_label.setText(value)
            
    def log_activity(self, message: str):
        """Log an activity message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.activity_log.append(f"[{timestamp}] {message}")


class VoxSigilProfessionalDashboard(QMainWindow):
    """Professional main dashboard window"""
    
    def __init__(self):
        super().__init__()
        self.vanta_core = None
        self.init_vanta_core()
        self.setup_ui()
        self.apply_professional_theme()
        
    def init_vanta_core(self):
        """Initialize VantaCore"""
        try:
            if UnifiedVantaCore:
                logger.info("Initializing VantaCore Professional...")
                self.vanta_core = UnifiedVantaCore()
                component_count = len(self.vanta_core.registry.list_components())
                logger.info(f"✅ VantaCore initialized with {component_count} components")
            else:
                logger.warning("VantaCore not available")
        except Exception as e:
            logger.error(f"Failed to initialize VantaCore: {e}")
            
    def setup_ui(self):
        """Setup the main UI"""
        self.setWindowTitle("🌟 VoxSigil Professional Training Dashboard")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Components gallery
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Header
        header_label = QLabel("🧩 VantaCore Components")
        header_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(header_label)
        
        # Components gallery
        self.components_gallery = ComponentsGalleryWidget(self.vanta_core)
        left_layout.addWidget(self.components_gallery)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Components")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        refresh_btn.clicked.connect(self.refresh_all)
        left_layout.addWidget(refresh_btn)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Status and controls
        self.status_panel = ProfessionalStatusPanel(self.vanta_core)
        splitter.addWidget(self.status_panel)
        
        # Set splitter proportions (70% components, 30% status)
        splitter.setSizes([1000, 400])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("🌟 VoxSigil Professional Dashboard - Ready")
        
        # Menu bar
        self.setup_menu()
        
    def setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        refresh_action = QAction('Refresh All', self)
        refresh_action.setShortcut('F5')
        refresh_action.triggered.connect(self.refresh_all)
        file_menu.addAction(refresh_action)
        
        file_menu.addSeparator()
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        fullscreen_action = QAction('Toggle Fullscreen', self)
        fullscreen_action.setShortcut('F11')
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def apply_professional_theme(self):
        """Apply professional theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QMenuBar {
                background-color: #343a40;
                color: white;
                border: none;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 8px;
            }
            QMenuBar::item:selected {
                background-color: #495057;
            }
            QMenu {
                background-color: white;
                border: 1px solid #dee2e6;
            }
            QStatusBar {
                background-color: #343a40;
                color: white;
                border: none;
            }
        """)
        
    def refresh_all(self):
        """Refresh all data"""
        logger.info("Refreshing all components...")
        self.components_gallery.refresh_components()
        self.status_panel.log_activity("Components refreshed")
        
        component_count = len(self.vanta_core.registry.list_components()) if self.vanta_core else 0
        self.statusBar().showMessage(f"🌟 Professional Dashboard - {component_count} components loaded - Refreshed")
        
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()
            
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About VoxSigil Professional Dashboard",
                         "VoxSigil Professional Training Dashboard v2.0\n\n"
                         "The ultimate streamlined interface for VantaCore\n"
                         "component management and AI training workflows.\n\n"
                         "✨ Features:\n"
                         "• Professional component gallery view\n"
                         "• Real-time system metrics dashboard\n"
                         "• Activity monitoring and logging\n"
                         "• Modern, responsive interface\n"
                         "• Complete VantaCore integration\n\n"
                         "Built with PyQt5 and VantaCore")


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("VoxSigil Professional Dashboard")
    app.setApplicationVersion("2.0")
    
    # Create and show window
    window = VoxSigilProfessionalDashboard()
    window.show()
    
    logger.info("🌟 VoxSigil Professional Dashboard launched successfully")
    
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
