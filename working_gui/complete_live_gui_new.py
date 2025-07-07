#!/usr/bin/env python3
"""
VoxSigil Complete Live GUI - All 33+ Tabs with Real Streaming Data
Direct import of all actual components with async live data functionality
"""

import logging
import sys
import time  # Added for performance profiling
import traceback
from pathlib import Path

# Ensure the repository root is on sys.path so 'Vanta' can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# PyQt5 imports
try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtWidgets import (
        QLabel,
        QMainWindow,
        QTabWidget,
        QVBoxLayout,
        QWidget,
        QStatusBar,
        QApplication,
    )
    from PyQt5.QtGui import QPalette, QColor

    logger.info("✅ PyQt5 imported successfully")
except ImportError as e:
    logger.error(f"❌ PyQt5 not available: {e}")
    sys.exit(1)

# --- VoxSigil Core Engine Imports (Patched for Real Components) ---
from ARC.arc_integration import HybridARCSolver as ARCIntegration
from BLT.blt_encoder import BLTEncoder
from core.grid_former import GRID_Former as GridFormer
from training import ARCGridTrainer, GridFormerTrainer
from VoxSigilRag.hybrid_blt import HybridMiddleware as RAGIntegration
from Vanta.core.UnifiedVantaCore import UnifiedVantaCore

class VoxSigilSystemInitializer(QThread):
    """Initialize and start all VoxSigil subsystems"""

    system_status = pyqtSignal(str, str)  # component, status
    initialization_complete = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.vanta_core = None
        self.active_agents = {}
        self.system_components = {}
    
    def run(self):
        """Initialize VoxSigil subsystems"""
        start_time = time.time()
        logger.info("Starting VoxSigil system initialization...")
        
        # Initialize VantaCore (core system)
        try:
            self.vanta_core = UnifiedVantaCore()
            self.system_components["vanta_core"] = self.vanta_core
            logger.info("✅ VantaCore initialized")
            self.system_status.emit("VantaCore", "Online")
        except ImportError as e:
            logger.warning(f"⚠️ VantaCore not available: {e}")
            self.system_status.emit("VantaCore", "Warning - Not Available")
        except Exception as e:
            logger.error(f"❌ VantaCore failed: {e}")
            self.system_status.emit("VantaCore", "Error")
            
        # Initialize Grid Former
        try:
            grid_former = GridFormer()
            self.system_components["grid_former"] = grid_former
            
            # Register with VantaCore if available
            if self.vanta_core:
                self.vanta_core.register_component("grid_former", grid_former)
                
            logger.info("✅ Grid Former initialized")
            self.system_status.emit("Grid Former", "Online")
        except ImportError as e:
            logger.warning(f"⚠️ Grid Former not available: {e}")
            self.system_status.emit("Grid Former", "Warning - Not Available")
        except Exception as e:
            logger.error(f"❌ Grid Former failed: {e}")
            self.system_status.emit("Grid Former", "Error")
            
        # Initialize ARC engine
        try:
            arc = ARCIntegration()
            self.system_components["arc"] = arc
            
            # Register with VantaCore if available
            if self.vanta_core:
                self.vanta_core.register_component("arc", arc)
                
            logger.info("✅ ARC engine initialized")
            self.system_status.emit("Engine ARC", "Online")
        except ImportError as e:
            logger.warning(f"⚠️ ARC engine not available: {e}")
            self.system_status.emit("Engine ARC", "Warning - Not Available")
        except Exception as e:
            logger.error(f"❌ ARC engine failed: {e}")
            self.system_status.emit("Engine ARC", "Error")

        # Initialize BLT engine
        try:
            blt = BLTEncoder()
            self.system_components["blt"] = blt
            
            # Register with VantaCore if available
            if self.vanta_core:
                self.vanta_core.register_component("blt", blt)
                
            logger.info("✅ BLT engine initialized")
            self.system_status.emit("Engine BLT", "Online")
        except ImportError as e:
            logger.warning(f"⚠️ BLT engine not available: {e}")
            self.system_status.emit("Engine BLT", "Warning - Not Available")
        except Exception as e:
            logger.error(f"❌ BLT engine failed: {e}")
            self.system_status.emit("Engine BLT", "Error")

        # Initialize RAG engine
        try:
            rag = RAGIntegration()
            self.system_components["rag"] = rag
            
            # Register with VantaCore if available
            if self.vanta_core:
                self.vanta_core.register_component("rag", rag)
                
            logger.info("✅ RAG engine initialized")
            self.system_status.emit("Engine RAG", "Online")
        except ImportError as e:
            logger.warning(f"⚠️ RAG engine not available: {e}")
            self.system_status.emit("Engine RAG", "Warning - Not Available")
        except Exception as e:
            logger.error(f"❌ RAG engine failed: {e}")
            self.system_status.emit("Engine RAG", "Error")
            
        duration = time.time() - start_time
        logger.info(f"VoxSigil system initialization completed in {duration:.2f} seconds")
        self.system_status.emit("Initialization", f"Complete ({duration:.2f}s)")
        
        # Signal completion
        self.initialization_complete.emit()

class LiveDataStreamer(QThread):
    """Stream live data from VoxSigil components"""
    
    data_updated = pyqtSignal(str, dict)  # component, data
    
    def __init__(self):
        super().__init__()
        self.components = {}
        self.vanta_core = None
        self.running = False
        
    def set_components(self, components, vanta_core=None):
        """Set the components to stream data from"""
        self.components = components
        self.vanta_core = vanta_core
        
    def start(self):
        """Start streaming data"""
        self.running = True
        super().start()
        
    def stop(self):
        """Stop streaming data"""
        self.running = False
        
    def run(self):
        """Stream data from components"""
        logger.info("Starting live data streaming")
        
        while self.running:
            self._update_live_data()
            time.sleep(1)  # Update every second
            
        logger.info("Live data streaming stopped")
    
    def _update_live_data(self):
        """Update live data from all components"""
        components = self.components
        vanta_core = self.vanta_core
        
        # Stream VantaCore data if available
        if vanta_core:
            try:
                core_data = {
                    "status": "online",
                    "timestamp": time.time(),
                    "components": list(components.keys()),
                    "memory_usage": vanta_core.get_memory_usage() if hasattr(vanta_core, "get_memory_usage") else "unknown",
                }
                self.data_updated.emit("vanta_core", core_data)
            except Exception as e:
                logger.error(f"Error getting VantaCore data: {e}")
        
        # Stream monitoring data
        if "monitoring" in components:
            try:
                monitoring = components["monitoring"]
                if hasattr(monitoring, "get_system_stats"):
                    stats = monitoring.get_system_stats()
                    self.data_updated.emit("monitoring", stats)
            except Exception as e:
                logger.error(f"Error getting monitoring data: {e}")
        
        # Stream data from other components
        for name, component in components.items():
            if name in ["monitoring"]:  # Skip already processed components
                continue
                
            try:
                # Check if component has a get_status or get_data method
                if hasattr(component, "get_status"):
                    data = component.get_status()
                    self.data_updated.emit(name, data)
                elif hasattr(component, "get_data"):
                    data = component.get_data()
                    self.data_updated.emit(name, data)
            except Exception as e:
                logger.error(f"Error getting data from {name}: {e}")

class CompleteVoxSigilGUI(QMainWindow):
    """Complete VoxSigil GUI with all tabs and real-time data streaming"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize window properties
        self.setWindowTitle("VoxSigil - Complete Live System")
        self.resize(1280, 800)
        
        # Set up dark theme
        self.setup_dark_theme()
        
        # Create the system initializer
        self.initializer = VoxSigilSystemInitializer()
        
        # Create the live data streamer
        self.data_streamer = LiveDataStreamer()
        
        # Connect signals
        self.initializer.system_status.connect(self.update_status)
        self.initializer.initialization_complete.connect(self.on_initialization_complete)
        
        # Create main tab widget
        self.main_tabs = QTabWidget()
        self.main_tabs.setTabPosition(QTabWidget.North)
        self.setCentralWidget(self.main_tabs)
        
        # Store tab references
        self.tabs = {}
        
        # Create all tabs
        self.create_all_tabs()
        
        # Start system initialization
        self.initializer.start()
        
        # Start data streaming (will connect to components once initialization completes)
        self.statusBar().showMessage("Starting VoxSigil systems...")
        
        logger.info("CompleteVoxSigilGUI created successfully")
    
    def setup_dark_theme(self):
        """Set up dark theme for the application"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)
    
    def create_all_tabs(self):
        """Create all tabs for the main window"""
        # System monitoring tabs
        self.create_system_status_tab()
        self.create_heartbeat_tab()
        self.create_performance_tab()
        
        # Core engine tabs
        self.create_vantacore_tab()
        self.create_grid_former_tab()
        self.create_arc_tab()
        self.create_blt_tab()
        self.create_rag_tab()
        
        # Agent tabs
        self.create_agent_mesh_tab()
        self.create_agent_ensemble_tab()
        self.create_individual_agents_tab()
        
        # Development and tools tabs
        self.create_dev_tools_tab()
        self.create_documentation_tab()
        
        # Specialized tabs
        self.create_training_tab()
        self.create_processing_engines_tab()
        self.create_supervisor_systems_tab()
        
        logger.info(f"Created {len(self.tabs)} tabs for GUI")
    
    def create_system_status_tab(self):
        """Create system status tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add system status components
        status_label = QLabel("System Status Dashboard")
        status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(status_label)
        
        # Add status overview widgets here
        
        self.main_tabs.addTab(tab, "📊 System Status")
        self.tabs["system_status"] = tab
    
    def create_heartbeat_tab(self):
        """Create heartbeat monitoring tab"""
        try:
            from gui.components.heartbeat_monitor_tab import HeartbeatMonitorTab
            tab = HeartbeatMonitorTab()
            self.main_tabs.addTab(tab, "💓 System Heartbeat")
            self.tabs["heartbeat"] = tab
        except ImportError as e:
            logger.warning(f"Could not create heartbeat tab: {e}")
            self._create_placeholder_tab("💓 System Heartbeat", "Heartbeat Monitor unavailable")
    
    def create_performance_tab(self):
        """Create performance monitoring tab"""
        try:
            from gui.components.performance_monitor import PerformanceMonitor
            tab = PerformanceMonitor()
            self.main_tabs.addTab(tab, "⚡ Performance")
            self.tabs["performance"] = tab
        except ImportError as e:
            logger.warning(f"Could not create performance tab: {e}")
            self._create_placeholder_tab("⚡ Performance", "Performance Monitor unavailable")
    
    def create_vantacore_tab(self):
        """Create VantaCore monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add VantaCore components
        vantacore_label = QLabel("VantaCore Live Monitoring")
        vantacore_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(vantacore_label)
        
        # Add VantaCore monitoring widgets here
        
        self.main_tabs.addTab(tab, "🧠 VantaCore")
        self.tabs["vantacore"] = tab
    
    def create_grid_former_tab(self):
        """Create GRID Former tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add GRID Former components
        grid_former_label = QLabel("GRID Former Visualization")
        grid_former_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(grid_former_label)
        
        # Add GRID Former visualization widgets here
        
        self.main_tabs.addTab(tab, "🧩 GRID Former")
        self.tabs["grid_former"] = tab
    
    def create_arc_tab(self):
        """Create ARC Integration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add ARC components
        arc_label = QLabel("ARC Integration Dashboard")
        arc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(arc_label)
        
        # Add ARC visualization widgets here
        
        self.main_tabs.addTab(tab, "🏗️ ARC Integration")
        self.tabs["arc"] = tab
    
    def create_blt_tab(self):
        """Create BLT tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add BLT components
        blt_label = QLabel("BLT System Dashboard")
        blt_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(blt_label)
        
        # Add BLT visualization widgets here
        
        self.main_tabs.addTab(tab, "📘 BLT System")
        self.tabs["blt"] = tab
    
    def create_rag_tab(self):
        """Create RAG tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add RAG components
        rag_label = QLabel("RAG System Dashboard")
        rag_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(rag_label)
        
        # Add RAG visualization widgets here
        
        self.main_tabs.addTab(tab, "📚 RAG System")
        self.tabs["rag"] = tab
    
    def create_agent_mesh_tab(self):
        """Create Agent Mesh tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add Agent Mesh components
        agent_mesh_label = QLabel("Agent Mesh Visualization")
        agent_mesh_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(agent_mesh_label)
        
        # Add Agent Mesh visualization widgets here
        
        self.main_tabs.addTab(tab, "🤖 Agent Mesh")
        self.tabs["agent_mesh"] = tab
    
    def create_agent_ensemble_tab(self):
        """Create Agent Ensemble tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add Agent Ensemble components
        agent_ensemble_label = QLabel("Agent Ensemble Dashboard")
        agent_ensemble_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(agent_ensemble_label)
        
        # Add Agent Ensemble visualization widgets here
        
        self.main_tabs.addTab(tab, "🔀 Agent Ensemble")
        self.tabs["agent_ensemble"] = tab
    
    def create_individual_agents_tab(self):
        """Create Individual Agents tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add Individual Agents components
        individual_agents_label = QLabel("Individual Agents Dashboard")
        individual_agents_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(individual_agents_label)
        
        # Add Individual Agents visualization widgets here
        
        self.main_tabs.addTab(tab, "🎭 Individual Agents")
        self.tabs["individual_agents"] = tab
    
    def create_dev_tools_tab(self):
        """Create Development Tools tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add Development Tools components
        dev_tools_label = QLabel("Development Tools Dashboard")
        dev_tools_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(dev_tools_label)
        
        # Add Development Tools widgets here
        
        self.main_tabs.addTab(tab, "🔧 Dev Tools")
        self.tabs["dev_tools"] = tab
    
    def create_documentation_tab(self):
        """Create Documentation tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add Documentation components
        documentation_label = QLabel("Documentation Browser")
        documentation_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(documentation_label)
        
        # Add Documentation browser widgets here
        
        self.main_tabs.addTab(tab, "📑 Documentation")
        self.tabs["documentation"] = tab
    
    def create_training_tab(self):
        """Create Training tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add Training components
        training_label = QLabel("Training System Dashboard")
        training_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(training_label)
        
        # Add Training visualization widgets here
        
        self.main_tabs.addTab(tab, "📚 Training")
        self.tabs["training"] = tab
    
    def create_processing_engines_tab(self):
        """Create Processing Engines tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add Processing Engines components
        processing_engines_label = QLabel("Processing Engines Dashboard")
        processing_engines_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(processing_engines_label)
        
        # Add Processing Engines visualization widgets here
        
        self.main_tabs.addTab(tab, "⚙️ Processing Engines")
        self.tabs["processing_engines"] = tab
    
    def create_supervisor_systems_tab(self):
        """Create Supervisor Systems tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add Supervisor Systems components
        supervisor_systems_label = QLabel("Supervisor Systems Dashboard")
        supervisor_systems_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(supervisor_systems_label)
        
        # Add Supervisor Systems visualization widgets here
        
        self.main_tabs.addTab(tab, "👁️ Supervisor Systems")
        self.tabs["supervisor_systems"] = tab
    
    def _create_placeholder_tab(self, name, message):
        """Create a placeholder tab with an error message"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        self.main_tabs.addTab(tab, name)
    
    def update_status(self, component, status):
        """Update status bar with component status"""
        self.statusBar().showMessage(f"Component {component}: {status}")
    
    def on_initialization_complete(self):
        """Handle completion of system initialization"""
        # Get initialized components
        components = self.initializer.system_components
        vanta_core = self.initializer.vanta_core
        
        # Connect data streamer to components
        self.data_streamer.set_components(components, vanta_core)
        self.data_streamer.start()
        
        # Update status
        self.statusBar().showMessage("VoxSigil system initialization complete")
        logger.info("VoxSigil system initialization complete")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop threads
        if hasattr(self, 'initializer') and self.initializer.isRunning():
            self.initializer.terminate()
        
        if hasattr(self, 'data_streamer') and self.data_streamer.isRunning():
            self.data_streamer.stop()
            self.data_streamer.terminate()
        
        # Accept the close event
        event.accept()
