"""
Optimized VoxSigil Complete Live GUI with Lazy Loading
This version uses lazy loading for tabs and components to improve startup time
"""

import logging
import sys
import threading
import time
import importlib

# Configure logging
logger = logging.getLogger("voxsigil.gui")

# PyQt5 imports
try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
    from PyQt5.QtWidgets import (
        QMainWindow,
        QTabWidget,
        QWidget,
        QVBoxLayout,
        QLabel,
        QPushButton,
    )
    from PyQt5.QtGui import QPalette, QColor
    
    # Import critical components for startup
    from gui.components.heartbeat_monitor_tab import HeartbeatMonitorTab
    from gui.components.status_display import SystemStatusDisplay
    
    logger.info("✅ PyQt5 imported successfully")
except ImportError as e:
    logger.error(f"Failed to import PyQt5: {e}")
    sys.exit(f"Error: PyQt5 is required to run this application. {e}")

# Custom Thread for system initialization
class SystemInitializationThread(QThread):
    """Thread for initializing system components in the background"""
    progress_signal = pyqtSignal(str, str)
    complete_signal = pyqtSignal()
    component_ready_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vanta_core = None
        self.active_agents = {}
        self.system_components = {}
        self.stop_flag = False

    def stop(self):
        """Stop the initialization thread"""
        self.stop_flag = True
        self.wait()

    def run(self):
        """Run the initialization sequence"""
        # 1. Initialize VantaCore
        self.progress_signal.emit("VantaCore", "Starting...")
        if not self._initialize_vanta_core():
            self.progress_signal.emit("VantaCore", "CRITICAL FAILURE")
        
        if self.stop_flag:
            return

        # 2. Start Agent Systems
        self.progress_signal.emit("Agents", "Starting...")
        self._start_agent_systems()
        
        if self.stop_flag:
            return

        # 3. Initialize Monitoring
        self.progress_signal.emit("Monitoring", "Starting...")
        self._initialize_monitoring()
        
        if self.stop_flag:
            return

        # 4. Start Training Systems
        self.progress_signal.emit("Training", "Starting...")
        self._initialize_training_systems()
        
        if self.stop_flag:
            return

        # 5. Initialize Processing Engines
        self.progress_signal.emit("Engines", "Starting...")
        self._initialize_processing_engines()
        
        self.progress_signal.emit("System", "Initialization sequence complete.")
        self.complete_signal.emit()

    def _initialize_vanta_core(self):
        """Initialize UnifiedVantaCore (full orchestration + cognitive engine)"""
        try:
            start_time = time.time()
            # Use importlib for more controlled importing
            module = importlib.import_module('Vanta.core.UnifiedVantaCore')
            UnifiedVantaCore = getattr(module, 'UnifiedVantaCore')
            
            self.vanta_core = UnifiedVantaCore()
            elapsed = time.time() - start_time
            logger.info(f"✅ UnifiedVantaCore initialized in {elapsed:.2f} seconds")
            self.progress_signal.emit("VantaCore", f"Online (Unified) - {elapsed:.2f}s")
            self.component_ready_signal.emit("vanta_core")
            return True
        except ImportError as e:
            logger.error(f"❌ UnifiedVantaCore not available: {e}")
            self.progress_signal.emit("VantaCore", "Error - Import Failed (Unified)")
            return False
        except Exception as e:
            logger.error(f"❌ UnifiedVantaCore initialization failed: {e}")
            self.progress_signal.emit("VantaCore", f"Error - {str(e)[:50]}")
            return False

    def _start_agent_systems(self):
        """Start all agent systems"""
        agent_list = [
            "andy", "astra", "oracle", "echo", "dreamer", "nebula", 
            "carla", "dave", "evo", "gizmo", "nix", "phi", "sam", "wendy"
        ]
        
        for agent_name in agent_list:
            if self.stop_flag:
                return
                
            try:
                start_time = time.time()
                # Try to import and initialize each agent
                module_name = f"agents.{agent_name}"
                agent_module = importlib.import_module(module_name)

                # Create agent instance
                class_name_options = [
                    f"{agent_name.capitalize()}Agent",
                    agent_name.capitalize(),
                ]
                agent_class = None
                for cn in class_name_options:
                    if hasattr(agent_module, cn):
                        agent_class = getattr(agent_module, cn)
                        break
                        
                if agent_class:
                    agent_instance = agent_class()
                    self.active_agents[agent_name] = agent_instance
                    elapsed = time.time() - start_time
                    logger.info(f"✅ Agent {agent_name} initialized in {elapsed:.2f}s")
                    self.progress_signal.emit(f"Agent {agent_name}", f"Online ({elapsed:.2f}s)")
                    self.component_ready_signal.emit(f"agent_{agent_name}")
                else:
                    logger.error(f"❌ Agent class for {agent_name} not found")
                    self.progress_signal.emit(f"Agent {agent_name}", "Error - Class Not Found")
            except Exception as e:
                logger.error(f"❌ Agent {agent_name} failed: {e}")
                self.progress_signal.emit(f"Agent {agent_name}", "Error")

        logger.info(f"Agent systems started: {len(self.active_agents)} agents active")

    def _initialize_monitoring(self):
        """Initialize monitoring systems"""
        if self.stop_flag:
            return
            
        try:
            start_time = time.time()
            module = importlib.import_module('monitoring.vanta_registration')
            MonitoringModule = getattr(module, 'MonitoringModule')
            
            monitoring = MonitoringModule()
            self.system_components["monitoring"] = monitoring
            elapsed = time.time() - start_time
            logger.info(f"✅ Monitoring systems initialized in {elapsed:.2f}s")
            self.progress_signal.emit("Monitoring", f"Online ({elapsed:.2f}s)")
            self.component_ready_signal.emit("monitoring")
        except Exception as e:
            logger.error(f"❌ Monitoring initialization failed: {e}")
            self.progress_signal.emit("Monitoring", "Error")

    def _initialize_training_systems(self):
        """Initialize training pipeline systems"""
        if self.stop_flag:
            return
            
        try:
            start_time = time.time()
            module = importlib.import_module('training.training_supervisor')
            TrainingSupervisor = getattr(module, 'TrainingSupervisor')
            
            training_sys = TrainingSupervisor()
            self.system_components["training"] = training_sys
            elapsed = time.time() - start_time
            logger.info(f"✅ Training systems initialized in {elapsed:.2f}s")
            self.progress_signal.emit("Training", f"Online ({elapsed:.2f}s)")
            self.component_ready_signal.emit("training")
        except Exception as e:
            logger.error(f"❌ Training systems initialization failed: {e}")
            self.progress_signal.emit("Training", "Warning - Not Available")

    def _initialize_processing_engines(self):
        """Initialize processing engines"""
        engines = [
            {
                "name": "gridformer",
                "module": "core.grid_former",
                "class": "GRID_Former",
                "display": "Engine GridFormer"
            },
            {
                "name": "arc",
                "module": "ARC.arc_integration",
                "class": "HybridARCSolver",
                "display": "Engine ARC"
            },
            {
                "name": "blt",
                "module": "BLT",
                "class": "BLTEncoder",
                "display": "Engine BLT"
            },
            {
                "name": "rag",
                "module": "VoxSigilRag.hybrid_blt",
                "class": "HybridMiddleware",
                "display": "Engine RAG"
            }
        ]
        
        for engine in engines:
            if self.stop_flag:
                return
                
            try:
                start_time = time.time()
                module = importlib.import_module(engine["module"])
                
                if engine["name"] == "blt":  # Special case for BLT
                    EngineClass = getattr(module, engine["class"])
                else:
                    EngineClass = getattr(module, engine["class"])
                
                instance = EngineClass()
                self.system_components[engine["name"]] = instance
                elapsed = time.time() - start_time
                logger.info(f"✅ {engine['display']} initialized in {elapsed:.2f}s")
                self.progress_signal.emit(engine["display"], f"Online ({elapsed:.2f}s)")
                self.component_ready_signal.emit(f"engine_{engine['name']}")
            except Exception as e:
                logger.error(f"❌ {engine['display']} failed: {e}")
                self.progress_signal.emit(engine["display"], "Warning - Not Available")


# Signal Handler class for live data streaming
class LiveDataStreamer(QObject):
    """Handler for live data streaming across tabs"""
    system_status = pyqtSignal(str, str)
    data_update = pyqtSignal(str, object)
    agent_status = pyqtSignal(str, str)
    engine_status = pyqtSignal(str, str)
    initialization_complete = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.update_thread = None
        self.stop_flag = threading.Event()
        
    def start_updates(self):
        """Start the live data update thread"""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.stop_flag.clear()
            self.update_thread = threading.Thread(target=self._update_live_data)
            self.update_thread.daemon = True
            self.update_thread.start()
            
    def stop_updates(self):
        """Stop the live data update thread"""
        if self.update_thread and self.update_thread.is_alive():
            self.stop_flag.set()
            self.update_thread.join(timeout=1.0)
            
    def _update_live_data(self):
        """Thread function to update live data"""
        while not self.stop_flag.is_set():
            # Emit signals with updated data
            # This will be handled by appropriate tabs
            time.sleep(1.0)  # Update interval


class LazyTabWidget(QWidget):
    """A placeholder widget that loads its real content on demand"""
    def __init__(self, parent=None, creator_func=None, tab_name=""):
        super().__init__(parent)
        self.creator_func = creator_func
        self.tab_name = tab_name
        self.real_widget = None
        self.initialized = False
        
        # Create placeholder layout
        layout = QVBoxLayout(self)
        self.placeholder_label = QLabel(f"Loading {tab_name}...")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.load_button = QPushButton("Load Now")
        self.load_button.clicked.connect(self.load_content)
        
        layout.addWidget(self.placeholder_label)
        layout.addWidget(self.load_button)
        
    def load_content(self):
        """Load the actual content of the tab"""
        if not self.initialized:
            try:
                start_time = time.time()
                self.placeholder_label.setText(f"Loading {self.tab_name}...")
                self.load_button.setEnabled(False)
                
                # Create the real widget
                self.real_widget = self.creator_func()
                
                # Replace placeholder with real widget
                layout = self.layout()
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
                
                layout.addWidget(self.real_widget)
                self.initialized = True
                
                elapsed = time.time() - start_time
                logger.info(f"Tab {self.tab_name} loaded in {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"Error loading tab {self.tab_name}: {e}")
                self.placeholder_label.setText(f"Error loading {self.tab_name}: {str(e)[:50]}...")
                self.load_button.setText("Retry Loading")
                self.load_button.setEnabled(True)
                
    def enterEvent(self, event):
        """Automatically load content when mouse hovers over tab"""
        if not self.initialized:
            self.load_content()
        super().enterEvent(event)


class CompleteVoxSigilGUI(QMainWindow):
    """Complete VoxSigil GUI with lazy-loaded tabs and components"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize window properties
        self.setWindowTitle("VoxSigil - Complete Interactive System")
        self.resize(1280, 800)
        
        # Set up dark theme
        self.setup_dark_theme()
        
        # Create the signal handler for live data
        self.signal_handler = LiveDataStreamer()
        
        # Create main tab widget
        self.main_tabs = QTabWidget()
        self.main_tabs.setTabPosition(QTabWidget.North)
        self.setCentralWidget(self.main_tabs)
        
        # Store tab references
        self.live_tabs = {}
        
        # Create system status display (always needed)
        self.status_display = SystemStatusDisplay()
        
        # Connect signals from data streamer to status display
        self.signal_handler.system_status.connect(self.status_display.update_status)
        self.signal_handler.agent_status.connect(self.status_display.update_agent_status)
        self.signal_handler.engine_status.connect(self.status_display.update_engine_status)
        
        # Create status bar
        self.statusBar().showMessage("Starting VoxSigil systems...")
        
        # Create critical tabs that should be loaded at startup
        self._create_critical_tabs()
        
        # Create placeholders for all other tabs
        self._create_lazy_loaded_tabs()
        
        # Set up initialization thread
        self.init_thread = SystemInitializationThread()
        self.init_thread.progress_signal.connect(self.signal_handler.system_status)
        self.init_thread.complete_signal.connect(self.signal_handler.initialization_complete)
        self.init_thread.complete_signal.connect(self.on_init_complete)
        
        # Start initialization in background
        self.init_thread.start()
        
        # Start live data updates
        self.signal_handler.start_updates()
        
        logger.info(f"GUI created with {self.main_tabs.count()} tabs (most lazy-loaded)")
        
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
        
    def on_init_complete(self):
        """Handle completion of initialization"""
        self.statusBar().showMessage("VoxSigil systems initialized successfully")
        
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop background threads
        self.init_thread.stop()
        self.signal_handler.stop_updates()
        event.accept()
        
    def _create_critical_tabs(self):
        """Create the critical tabs that should be loaded at startup"""
        # System status tab is always needed
        status_tab = self._create_system_status_tab()
        self.main_tabs.addTab(status_tab, "📊 System Status")
        self.live_tabs["System Status"] = status_tab
        
        # Heartbeat monitor is lightweight and useful
        heartbeat_tab = self._create_heartbeat_tab()
        self.main_tabs.addTab(heartbeat_tab, "💓 System Heartbeat")
        self.live_tabs["System Heartbeat"] = heartbeat_tab
        
    def _create_lazy_loaded_tabs(self):
        """Create placeholders for all non-critical tabs"""
        lazy_tabs = [
            ("🤖 Agent Mesh", self._create_agent_mesh_tab),
            ("🧠 VantaCore", self._create_vantacore_tab),
            ("⚡ Performance", self._create_performance_tab),
            ("🔄 Live Streaming", self._create_streaming_tab),
            ("🎭 Individual Agents", self._create_individual_agents_tab),
            ("🌐 Agent Networks", self._create_agent_networks_tab),
            ("🔀 Agent Ensemble", self._create_agent_ensemble_tab),
            ("💬 Communication", self._create_communication_tab),
            ("📋 Task Queue", self._create_task_queue_tab),
            ("🧩 GRID Former", self._create_grid_former_tab),
            ("🏗️ ARC Integration", self._create_arc_integration_tab),
            ("📚 Training", self._create_training_tab),
            ("📘 BLT/RAG Systems", self._create_blt_rag_tab),
            ("🔍 Inference", self._create_inference_tab),
            ("📊 System Resources", self._create_system_resources_tab),
            ("📈 Performance Metrics", self._create_performance_metrics_tab),
            ("📉 Error Analytics", self._create_error_analytics_tab),
            ("🔔 Alerts", self._create_alerts_tab),
            ("⚙️ Configuration", self._create_configuration_tab),
            ("🛠️ Developer Tools", self._create_developer_tools_tab),
            ("📝 API Browser", self._create_api_browser_tab),
            ("🧪 Testing", self._create_testing_tab),
            ("🐞 Debugging", self._create_debugging_tab),
            ("🎯 Tuning", self._create_tuning_tab),
            ("🔮 Simulation", self._create_simulation_tab),
            ("🔧 System Tools", self._create_system_tools_tab),
            ("📱 Mobile Interface", self._create_mobile_interface_tab),
            ("📡 Network Diagnostics", self._create_network_diagnostics_tab),
            ("🔒 Security", self._create_security_tab),
            ("📑 Documentation", self._create_documentation_tab),
        ]
        
        for tab_name, creator_func in lazy_tabs:
            lazy_widget = LazyTabWidget(creator_func=creator_func, tab_name=tab_name)
            self.main_tabs.addTab(lazy_widget, tab_name)
            self.live_tabs[tab_name] = lazy_widget
            
    # Critical tab creation methods
    def _create_system_status_tab(self):
        """Create system status overview tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(self.status_display)
        return tab
        
    def _create_heartbeat_tab(self):
        """Create heartbeat monitoring tab"""
        return HeartbeatMonitorTab()
        
    # Lazy-loaded tab creation methods
    def _create_agent_mesh_tab(self):
        """Create agent mesh visualization tab"""
        from gui.components.agent_mesh_visualizer import AgentMeshVisualizer
        return AgentMeshVisualizer()
        
    def _create_vantacore_tab(self):
        """Create VantaCore monitoring tab"""
        from gui.components.vantacore_monitor import VantaCoreMonitor
        return VantaCoreMonitor(vanta_core=self.init_thread.vanta_core)
        
    def _create_performance_tab(self):
        """Create performance monitoring tab"""
        from gui.components.performance_monitor import PerformanceMonitor
        return PerformanceMonitor()
        
    def _create_streaming_tab(self):
        """Create live streaming data tab"""
        from gui.components.streaming_monitor import StreamingMonitor
        return StreamingMonitor()
        
    def _create_individual_agents_tab(self):
        """Create individual agents tab"""
        from gui.components.individual_agents import IndividualAgentsTab
        return IndividualAgentsTab(agents=self.init_thread.active_agents)
        
    def _create_agent_networks_tab(self):
        """Create agent networks tab"""
        from gui.components.agent_networks import AgentNetworksTab
        return AgentNetworksTab()
        
    def _create_agent_ensemble_tab(self):
        """Create agent ensemble tab"""
        from gui.components.agent_ensemble import AgentEnsembleTab
        return AgentEnsembleTab()
        
    def _create_communication_tab(self):
        """Create communication tab"""
        from gui.components.communication_monitor import CommunicationMonitor
        return CommunicationMonitor()
        
    def _create_task_queue_tab(self):
        """Create task queue tab"""
        from gui.components.task_queue import TaskQueueTab
        return TaskQueueTab()
        
    def _create_grid_former_tab(self):
        """Create GRID Former tab"""
        from gui.components.grid_former_monitor import GridFormerMonitor
        gridformer = self.init_thread.system_components.get("gridformer")
        return GridFormerMonitor(gridformer=gridformer)
        
    def _create_arc_integration_tab(self):
        """Create ARC integration tab"""
        from gui.components.arc_integration_tab import ARCIntegrationTab
        arc = self.init_thread.system_components.get("arc")
        return ARCIntegrationTab(arc=arc)
        
    def _create_training_tab(self):
        """Create training tab"""
        from gui.components.training_monitor import TrainingMonitor
        training = self.init_thread.system_components.get("training")
        return TrainingMonitor(training=training)
        
    def _create_blt_rag_tab(self):
        """Create BLT/RAG systems tab"""
        from gui.components.blt_rag_systems import BLTRAGSystemsTab
        blt = self.init_thread.system_components.get("blt")
        rag = self.init_thread.system_components.get("rag")
        return BLTRAGSystemsTab(blt=blt, rag=rag)
        
    def _create_inference_tab(self):
        """Create inference tab"""
        from gui.components.inference_monitor import InferenceMonitor
        return InferenceMonitor()
        
    def _create_system_resources_tab(self):
        """Create system resources tab"""
        from gui.components.system_resources import SystemResourcesTab
        return SystemResourcesTab()
        
    def _create_performance_metrics_tab(self):
        """Create performance metrics tab"""
        from gui.components.performance_metrics import PerformanceMetricsTab
        return PerformanceMetricsTab()
        
    def _create_error_analytics_tab(self):
        """Create error analytics tab"""
        from gui.components.error_analytics import ErrorAnalyticsTab
        return ErrorAnalyticsTab()
        
    def _create_alerts_tab(self):
        """Create alerts tab"""
        from gui.components.alerts import AlertsTab
        return AlertsTab()
        
    def _create_configuration_tab(self):
        """Create configuration tab"""
        from gui.components.configuration import ConfigurationTab
        return ConfigurationTab()
        
    def _create_developer_tools_tab(self):
        """Create developer tools tab"""
        from gui.components.developer_tools import DeveloperToolsTab
        return DeveloperToolsTab()
        
    def _create_api_browser_tab(self):
        """Create API browser tab"""
        from gui.components.api_browser import APIBrowserTab
        return APIBrowserTab()
        
    def _create_testing_tab(self):
        """Create testing tab"""
        from gui.components.testing import TestingTab
        return TestingTab()
        
    def _create_debugging_tab(self):
        """Create debugging tab"""
        from gui.components.debugging import DebuggingTab
        return DebuggingTab()
        
    def _create_tuning_tab(self):
        """Create tuning tab"""
        from gui.components.tuning import TuningTab
        return TuningTab()
        
    def _create_simulation_tab(self):
        """Create simulation tab"""
        from gui.components.simulation import SimulationTab
        return SimulationTab()
        
    def _create_system_tools_tab(self):
        """Create system tools tab"""
        from gui.components.system_tools import SystemToolsTab
        return SystemToolsTab()
        
    def _create_mobile_interface_tab(self):
        """Create mobile interface tab"""
        from gui.components.mobile_interface import MobileInterfaceTab
        return MobileInterfaceTab()
        
    def _create_network_diagnostics_tab(self):
        """Create network diagnostics tab"""
        from gui.components.network_diagnostics import NetworkDiagnosticsTab
        return NetworkDiagnosticsTab()
        
    def _create_security_tab(self):
        """Create security tab"""
        from gui.components.security import SecurityTab
        return SecurityTab()
        
    def _create_documentation_tab(self):
        """Create documentation tab"""
        from gui.components.documentation import DocumentationTab
        return DocumentationTab()

# QObject for LiveDataStreamer
class QObject(object):
    """Base QObject class"""
    def __init__(self):
        pass
class QObject(object):
    """Base QObject class - placeholder if PyQt import fails"""
    def __init__(self):
        pass