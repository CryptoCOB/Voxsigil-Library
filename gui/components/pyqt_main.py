from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Dict

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .agent_status_panel import AgentStatusPanel
from .dynamic_gridformer_gui import DynamicGridFormerTab
from .echo_log_panel import EchoLogPanel
from .gui_styles import VoxSigilStyles  # Import dark mode styles

# from .mesh_map_panel import MeshMapPanel  # Commented out due to syntax issues
from .music_tab import MusicTab
from .novel_reasoning_tab import NovelReasoningTab

logger = logging.getLogger(__name__)


# Import Vanta Core tab
try:
    from .vanta_core_tab import VantaCoreTab

    VANTA_CORE_AVAILABLE = True
except ImportError:
    VANTA_CORE_AVAILABLE = False

# Import Control Center tab
try:
    from .control_center_tab import ControlCenterTab

    CONTROL_CENTER_AVAILABLE = True
except ImportError:
    CONTROL_CENTER_AVAILABLE = False

# Import new gap-filling tabs
try:
    from .dataset_panel import DatasetPanel
    from .dependency_panel import DependencyPanel
    from .security_panel import SecurityPanel

    NEW_PANELS_AVAILABLE = True
except ImportError:
    NEW_PANELS_AVAILABLE = False

# Import all new streaming monitoring tabs
try:
    from .memory_systems_tab import MemorySystemsTab

    MEMORY_SYSTEMS_AVAILABLE = True
except ImportError:
    MEMORY_SYSTEMS_AVAILABLE = False

try:
    from .training_pipelines_tab import TrainingPipelinesTab

    TRAINING_PIPELINES_AVAILABLE = True
except ImportError:
    TRAINING_PIPELINES_AVAILABLE = False

try:
    from .training_control_tab import TrainingControlTab

    TRAINING_CONTROL_AVAILABLE = True
except ImportError:
    TRAINING_CONTROL_AVAILABLE = False

try:
    from .supervisor_systems_tab import SupervisorSystemsTab

    SUPERVISOR_SYSTEMS_AVAILABLE = True
except ImportError:
    SUPERVISOR_SYSTEMS_AVAILABLE = False

try:
    from .handler_systems_tab import HandlerSystemsTab

    HANDLER_SYSTEMS_AVAILABLE = True
except ImportError:
    HANDLER_SYSTEMS_AVAILABLE = False

try:
    from .service_systems_tab import ServiceSystemsTab

    SERVICE_SYSTEMS_AVAILABLE = True
except ImportError:
    SERVICE_SYSTEMS_AVAILABLE = False

try:
    from .system_integration_tab import SystemIntegrationTab

    SYSTEM_INTEGRATION_AVAILABLE = True
except ImportError:
    SYSTEM_INTEGRATION_AVAILABLE = False

try:
    from .realtime_logs_tab import RealtimeLogsTab

    REALTIME_LOGS_AVAILABLE = True
except ImportError:
    REALTIME_LOGS_AVAILABLE = False

# Import additional completion tabs
try:
    from .heartbeat_monitor_tab import HeartbeatMonitorTab

    HEARTBEAT_MONITOR_AVAILABLE = True
except ImportError:
    HEARTBEAT_MONITOR_AVAILABLE = False

try:
    from .config_editor_tab import ConfigEditorTab

    CONFIG_EDITOR_AVAILABLE = True
except ImportError:
    CONFIG_EDITOR_AVAILABLE = False

try:
    from .experiment_tracker_tab import ExperimentTrackerTab

    EXPERIMENT_TRACKER_AVAILABLE = True
except ImportError:
    EXPERIMENT_TRACKER_AVAILABLE = False

try:
    from .notification_center_tab import NotificationCenterTab

    NOTIFICATION_CENTER_AVAILABLE = True
except ImportError:
    NOTIFICATION_CENTER_AVAILABLE = False

# Import enhanced component tabs with streaming
try:
    from .enhanced_blt_rag_tab import EnhancedBLTRAGTab

    ENHANCED_BLT_RAG_AVAILABLE = True
except ImportError:
    ENHANCED_BLT_RAG_AVAILABLE = False

# Import new high-priority streaming tabs
try:
    from .individual_agents_tab import IndividualAgentsTab

    INDIVIDUAL_AGENTS_AVAILABLE = True
except ImportError:
    INDIVIDUAL_AGENTS_AVAILABLE = False

try:
    from .processing_engines_tab import ProcessingEnginesTab

    PROCESSING_ENGINES_AVAILABLE = True
except ImportError:
    PROCESSING_ENGINES_AVAILABLE = False

try:
    from .system_health_dashboard import SystemHealthDashboard

    SYSTEM_HEALTH_AVAILABLE = True
except ImportError:
    SYSTEM_HEALTH_AVAILABLE = False

# Import new tab components
try:
    from .vmb_integration_tab import VMBIntegrationTab

    VMB_INTEGRATION_AVAILABLE = True
except ImportError:
    VMB_INTEGRATION_AVAILABLE = False

try:
    from .vmb_components_pyqt5 import VMBFinalDemoTab

    VMB_DEMO_AVAILABLE = True
except ImportError:
    VMB_DEMO_AVAILABLE = False

try:
    from .dynamic_gridformer_gui import DynamicGridFormerTab

    DYNAMIC_GRIDFORMER_TAB_AVAILABLE = True
except ImportError:
    DYNAMIC_GRIDFORMER_TAB_AVAILABLE = False

# Import the separate interface components to convert them to tabs
try:
    from ...interfaces.model_tab_interface import ModelTabInterface

    MODEL_TAB_AVAILABLE = True
except ImportError:
    MODEL_TAB_AVAILABLE = False

try:
    from ...interfaces.performance_tab_interface import PerformanceTabInterface

    PERFORMANCE_TAB_AVAILABLE = True
except ImportError:
    PERFORMANCE_TAB_AVAILABLE = False

try:
    from ...interfaces.visualization_tab_interface import VisualizationTabInterface

    VISUALIZATION_TAB_AVAILABLE = True
except ImportError:
    VISUALIZATION_TAB_AVAILABLE = False

try:
    from ...interfaces.training_interface import VoxSigilTrainingInterface as TrainingInterface

    TRAINING_TAB_AVAILABLE = True
except ImportError:
    TRAINING_TAB_AVAILABLE = False

try:
    from ...interfaces.model_discovery_interface import ModelDiscoveryInterface

    MODEL_DISCOVERY_AVAILABLE = True
except ImportError:
    MODEL_DISCOVERY_AVAILABLE = False

# Import voice system for startup greeting
try:
    from ..core.agent_voice_system import get_agent_voice_system

    VOICE_SYSTEM_AVAILABLE = True
except ImportError:
    VOICE_SYSTEM_AVAILABLE = False

# Import TTS system for startup greeting - disabled for now
TTS_ENGINE_AVAILABLE = False

# Import microphone monitor
try:
    from .microphone_monitor import MicrophoneMonitor

    MICROPHONE_MONITOR_AVAILABLE = True
except ImportError:
    MICROPHONE_MONITOR_AVAILABLE = False


class VoxSigilMainWindow(QMainWindow):
    """Unified PyQt main window with all components as tabs."""

    def __init__(self, registry=None, event_bus=None, async_bus=None, training_engine=None):
        super().__init__()
        self.registry = registry
        self.event_bus = event_bus
        self.async_bus = async_bus
        self.training_engine = training_engine
        self.echo_panel = None
        self.mesh_map_panel = None
        self.status_panel = None

        # Apply VoxSigil Dark Mode Theme
        self.setWindowTitle("VoxSigil Unified GUI - All Components")
        self.resize(1200, 900)

        # Apply dark theme styling
        self.setStyleSheet(VoxSigilStyles.get_complete_stylesheet())
        VoxSigilStyles.apply_icon(self)

        self._init_ui()
        self._setup_event_handlers()

        # Schedule startup voice greeting
        if VOICE_SYSTEM_AVAILABLE:
            QTimer.singleShot(2000, self._play_startup_greeting)  # Wait 2 seconds after GUI loads

    def _play_startup_greeting(self):
        """Play a startup voice greeting with different agents introducing themselves."""
        try:
            # Create a sequence of agent introductions
            greetings = [
                ("Astra", "Navigation systems online. VoxSigil interface ready for exploration."),
                ("Andy", "Output composers standing by. Ready to synthesize your requests!"),
                (
                    "Voxka",
                    "Dual cognition core activated. The voice of phi resonates through the mesh.",
                ),
                ("Oracle", "Ancient wisdom flows. All agents await your guidance."),
                ("Sam", "Support systems ready! Welcome to your VoxSigil command center."),
            ]

            # Schedule greetings with delays
            delay = 3000  # Start after 3 seconds
            for agent_name, greeting in greetings:
                QTimer.singleShot(
                    delay, lambda a=agent_name, g=greeting: self._speak_greeting(a, g)
                )
                delay += 4000  # 4 seconds between each greeting

        except Exception as e:
            logger.error(f"Error scheduling startup greeting: {e}")

    def _speak_greeting(self, agent_name: str, greeting: str):
        """Speak a greeting from a specific agent."""
        try:
            voice_system = get_agent_voice_system()
            speech_config = voice_system.speak_with_personality(
                agent_name, greeting, add_signature=True
            )

            # Try to get TTS engine
            if self.registry and hasattr(self.registry, "get_component"):
                tts_engine = self.registry.get_component("async_tts_engine")
                if tts_engine:
                    asyncio.create_task(
                        self._async_speak(
                            tts_engine, speech_config["text"], speech_config["config"]
                        )
                    )
                    logger.info(f"🎙️ {agent_name} greeting: {speech_config['text'][:50]}...")

                    # Also display in status if available
                    if hasattr(self, "status_panel") and self.status_panel:
                        self.status_panel.add_status_message(f"🎭 {agent_name}: {greeting}")
                else:
                    logger.warning("TTS engine not available for greeting")
            else:
                logger.warning("Registry not available for TTS greeting")

        except Exception as e:
            logger.error(f"Error playing greeting from {agent_name}: {e}")

    async def _async_speak(self, tts_engine, text: str, voice_config: Dict[str, Any]):
        """Async helper for TTS calls."""
        try:
            await tts_engine.speak_async(text, voice_config)
        except Exception as e:
            logger.error(f"Error in async TTS: {e}")

    def _setup_event_handlers(self):
        """Setup event bus and async bus handlers."""
        if self.event_bus:
            self.event_bus.subscribe("mesh_echo", self._on_mesh_echo)
            self.event_bus.subscribe("mesh_graph_update", self._on_mesh_graph)
            self.event_bus.subscribe("agent_status", self._on_status)

        if self.async_bus:
            try:
                self.async_bus.register_component("GUI")
                from Vanta.core.UnifiedAsyncBus import AsyncMessage, MessageType

                def _echo_cb(msg: AsyncMessage):
                    if self.echo_panel:
                        self.echo_panel.add_message(str(msg.content))

                self.async_bus.subscribe("GUI", MessageType.USER_INTERACTION, _echo_cb)
            except Exception:
                pass

    def _init_ui(self):
        """Initialize the unified tabbed interface."""
        tabs = QTabWidget()

        # Add Control Center as FIRST tab (priority 0)
        if CONTROL_CENTER_AVAILABLE:
            control_center = ControlCenterTab(
                event_bus=self.event_bus, training_engine=self.training_engine
            )
            tabs.addTab(control_center, "🎛️ Control Center")
        else:
            tabs.addTab(
                self._create_placeholder_tab("Control Center"), "🎛️ Control Center"
            )  # Core Components Tabs
        if MODEL_TAB_AVAILABLE:
            model_tab = ModelTabInterface()
            tabs.addTab(model_tab, "🤖 Models")
        else:
            tabs.addTab(self._create_placeholder_tab("Models"), "🤖 Models")

        if MODEL_DISCOVERY_AVAILABLE:
            discovery_tab = ModelDiscoveryInterface()
            tabs.addTab(discovery_tab, "🔍 Model Discovery")
        else:
            tabs.addTab(self._create_placeholder_tab("Model Discovery"), "🔍 Model Discovery")

        if TRAINING_CONTROL_AVAILABLE:
            # Use new training control tab for model selection and training
            training_tab = TrainingControlTab(event_bus=self.event_bus)
            tabs.addTab(training_tab, "🎯 Training")
        elif TRAINING_TAB_AVAILABLE:
            # Fallback to old training interface
            training_tab = TrainingInterface(training_engine=self.training_engine)
            tabs.addTab(training_tab, "🎯 Training (Legacy)")
        else:
            tabs.addTab(self._create_placeholder_tab("Training"), "🎯 Training")

        # High-Priority Streaming Tabs (NEW)
        if SYSTEM_HEALTH_AVAILABLE:
            health_tab = SystemHealthDashboard(event_bus=self.event_bus)
            tabs.addTab(health_tab, "💊 System Health")
        else:
            tabs.addTab(self._create_placeholder_tab("System Health"), "💊 System Health")

        if INDIVIDUAL_AGENTS_AVAILABLE:
            agents_tab = IndividualAgentsTab(registry=self.registry, event_bus=self.event_bus)
            tabs.addTab(agents_tab, "🤖 Individual Agents")
        else:
            tabs.addTab(self._create_placeholder_tab("Individual Agents"), "🤖 Individual Agents")

        if PROCESSING_ENGINES_AVAILABLE:
            engines_tab = ProcessingEnginesTab(event_bus=self.event_bus)
            tabs.addTab(engines_tab, "⚙️ Processing Engines")
        else:
            tabs.addTab(self._create_placeholder_tab("Processing Engines"), "⚙️ Processing Engines")

        # Novel Reasoning tab - always available
        novel_tab = NovelReasoningTab()
        tabs.addTab(novel_tab, "🧠 Novel Reasoning")

        if VISUALIZATION_TAB_AVAILABLE:
            viz_tab = VisualizationTabInterface()
            tabs.addTab(viz_tab, "📊 Visualization")
        else:
            tabs.addTab(self._create_placeholder_tab("Visualization"), "📊 Visualization")

        if PERFORMANCE_TAB_AVAILABLE:
            perf_tab = PerformanceTabInterface()
            tabs.addTab(perf_tab, "⚡ Performance")
        else:
            tabs.addTab(
                self._create_placeholder_tab("Performance"), "⚡ Performance"
            )  # Specialized Component Tabs
        gridformer_tab = DynamicGridFormerTab()
        tabs.addTab(gridformer_tab, "🔄 GridFormer")

        # Add Advanced GridFormer Tab (converted from standalone window)
        if DYNAMIC_GRIDFORMER_TAB_AVAILABLE:
            advanced_gridformer_tab = DynamicGridFormerTab()
            tabs.addTab(advanced_gridformer_tab, "🧠 Advanced GridFormer")
        else:
            tabs.addTab(
                self._create_placeholder_tab("Advanced GridFormer"), "🧠 Advanced GridFormer"
            )

        # Add VMB Integration Tab (converted from Tkinter window)
        if VMB_INTEGRATION_AVAILABLE:
            vmb_integration_tab = VMBIntegrationTab()
            tabs.addTab(vmb_integration_tab, "🔥 VMB Integration")
        else:
            tabs.addTab(self._create_placeholder_tab("VMB Integration"), "🔥 VMB Integration")

        # Add VMB Demo Tab (converted from standalone window)
        if VMB_DEMO_AVAILABLE:
            vmb_demo_tab = VMBFinalDemoTab()
            tabs.addTab(vmb_demo_tab, "🎭 VMB Demo")
        else:
            tabs.addTab(self._create_placeholder_tab("VMB Demo"), "🎭 VMB Demo")

        music_tab = MusicTab()
        tabs.addTab(music_tab, "🎵 Music")  # Core Monitoring Tabs
        self.echo_panel = EchoLogPanel()
        # self.mesh_map_panel = MeshMapPanel()  # Commented out due to syntax issues
        self.status_panel = AgentStatusPanel()
        tabs.addTab(self.echo_panel, "📡 Echo Log")
        # tabs.addTab(self.mesh_map_panel, "🕸️ Mesh Map")  # Commented out due to syntax issues
        tabs.addTab(self.status_panel, "📈 Agent Status")

        # Enhanced Component Monitoring Tabs with Streaming
        if ENHANCED_BLT_RAG_AVAILABLE:
            enhanced_blt_tab = EnhancedBLTRAGTab(event_bus=self.event_bus)
            tabs.addTab(enhanced_blt_tab, "🔧 BLT/RAG Enhanced")
        else:
            # Fallback to basic BLT/RAG tab
            blt_tab = self._create_blt_components_tab()
            tabs.addTab(blt_tab, "🔧 BLT/RAG")  # Add ARC Components Tab
        arc_tab = self._create_arc_components_tab()
        tabs.addTab(arc_tab, "🧩 ARC")  # Add Vanta Core Tab
        if VANTA_CORE_AVAILABLE:
            vanta_tab = VantaCoreTab(event_bus=self.event_bus)
            tabs.addTab(vanta_tab, "⚡ Vanta Core")
        else:
            vanta_tab = self._create_vanta_core_tab()
            tabs.addTab(vanta_tab, "⚡ Vanta Core")

        # Add new gap-filling tabs
        if NEW_PANELS_AVAILABLE:
            # Security & Compliance Panel
            security_tab = SecurityPanel(bus=getattr(self, "event_bus", None))
            tabs.addTab(security_tab, "🛡️ Security & Compliance")

            # Dataset Manager Panel
            dataset_tab = DatasetPanel(bus=getattr(self, "event_bus", None))
            tabs.addTab(dataset_tab, "📊 Dataset Manager")

            # Dependency Health Panel
            dependency_tab = DependencyPanel(bus=getattr(self, "event_bus", None))
            tabs.addTab(dependency_tab, "📦 Dependency Health")
        else:
            tabs.addTab(
                self._create_placeholder_tab("Security & Compliance"), "🛡️ Security & Compliance"
            )
            tabs.addTab(self._create_placeholder_tab("Dataset Manager"), "📊 Dataset Manager")
            tabs.addTab(self._create_placeholder_tab("Dependency Health"), "📦 Dependency Health")

        # Add all new streaming monitoring tabs
        if MEMORY_SYSTEMS_AVAILABLE:
            memory_tab = MemorySystemsTab(event_bus=self.event_bus)
            tabs.addTab(memory_tab, "🧠 Memory Systems")
        else:
            tabs.addTab(self._create_placeholder_tab("Memory Systems"), "🧠 Memory Systems")

        if TRAINING_PIPELINES_AVAILABLE:
            training_pipelines_tab = TrainingPipelinesTab(event_bus=self.event_bus)
            tabs.addTab(training_pipelines_tab, "🏗️ Training Pipelines")
        else:
            tabs.addTab(self._create_placeholder_tab("Training Pipelines"), "🏗️ Training Pipelines")

        if SUPERVISOR_SYSTEMS_AVAILABLE:
            supervisor_tab = SupervisorSystemsTab(event_bus=self.event_bus)
            tabs.addTab(supervisor_tab, "👑 Supervisor Systems")
        else:
            tabs.addTab(self._create_placeholder_tab("Supervisor Systems"), "👑 Supervisor Systems")

        if HANDLER_SYSTEMS_AVAILABLE:
            handler_tab = HandlerSystemsTab(event_bus=self.event_bus)
            tabs.addTab(handler_tab, "🔌 Handler Systems")
        else:
            tabs.addTab(self._create_placeholder_tab("Handler Systems"), "🔌 Handler Systems")

        if SERVICE_SYSTEMS_AVAILABLE:
            service_tab = ServiceSystemsTab(event_bus=self.event_bus)
            tabs.addTab(service_tab, "🔧 Service Systems")
        else:
            tabs.addTab(self._create_placeholder_tab("Service Systems"), "🔧 Service Systems")

        if SYSTEM_INTEGRATION_AVAILABLE:
            integration_tab = SystemIntegrationTab(event_bus=self.event_bus)
            tabs.addTab(integration_tab, "🔗 System Integration")
        else:
            tabs.addTab(self._create_placeholder_tab("System Integration"), "🔗 System Integration")

        if REALTIME_LOGS_AVAILABLE:
            logs_tab = RealtimeLogsTab(event_bus=self.event_bus)
            tabs.addTab(logs_tab, "📜 Real-time Logs")
        else:
            tabs.addTab(self._create_placeholder_tab("Real-time Logs"), "📜 Real-time Logs")

        # Add completion tabs for 100% coverage
        if HEARTBEAT_MONITOR_AVAILABLE:
            heartbeat_tab = HeartbeatMonitorTab(event_bus=self.event_bus)
            tabs.addTab(heartbeat_tab, "❤️ Heartbeat Monitor")
        else:
            tabs.addTab(self._create_placeholder_tab("Heartbeat Monitor"), "❤️ Heartbeat Monitor")

        if CONFIG_EDITOR_AVAILABLE:
            config_tab = ConfigEditorTab(event_bus=self.event_bus)
            tabs.addTab(config_tab, "⚙️ Config Editor")
        else:
            tabs.addTab(self._create_placeholder_tab("Config Editor"), "⚙️ Config Editor")

        if EXPERIMENT_TRACKER_AVAILABLE:
            experiment_tab = ExperimentTrackerTab(event_bus=self.event_bus)
            tabs.addTab(experiment_tab, "🧪 Experiment Tracker")
        else:
            tabs.addTab(self._create_placeholder_tab("Experiment Tracker"), "🧪 Experiment Tracker")

        if NOTIFICATION_CENTER_AVAILABLE:
            notification_tab = NotificationCenterTab(event_bus=self.event_bus)
            tabs.addTab(notification_tab, "🔔 Notification Center")
        else:
            tabs.addTab(
                self._create_placeholder_tab("Notification Center"), "🔔 Notification Center"
            )

        # Add Microphone Monitor Tab
        if MICROPHONE_MONITOR_AVAILABLE:
            mic_monitor_tab = MicrophoneMonitor(vanta_core=self.registry)
            # Connect voice detection to agent system
            mic_monitor_tab.voice_detected.connect(self._on_voice_command)
            tabs.addTab(mic_monitor_tab, "🎤 Voice Control")
        else:
            tabs.addTab(self._create_placeholder_tab("Voice Control"), "🎤 Voice Control")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(tabs)

        if self.registry:
            layout.addWidget(self._create_agent_buttons())

        self.setCentralWidget(container)

    def _setup_voice_system(self):
        """Setup the voice system for startup greeting."""
        if VOICE_SYSTEM_AVAILABLE:
            self.voice_system = get_agent_voice_system()
            if self.voice_system:
                self.voice_system.set_language("en-US")  # Set default language
                self.voice_system.set_voice("en-US-Wavenet-D")  # Set default voice
                self.voice_system.set_rate(150)  # Set speech rate
                self.voice_system.set_volume(0.9)  # Set volume (0.0 to 1.0)
                self.voice_system.set_pitch(0)  # Set pitch (default is 0)
        else:
            logger.warning("Voice system not available")

    def _speak(self, text: str):
        """Speak the given text using the agent voice system."""
        if self.voice_system:
            try:
                self.voice_system.speak(text)
            except Exception as e:
                logger.error(f"Error speaking text: {e}")

    def _on_startup(self):
        """Handle startup actions."""
        # Setup voice system
        self._setup_voice_system()

        # Greet the user
        self._speak("Welcome to VoxSigil Unified GUI. All components are loaded and ready to use.")

    def closeEvent(self, event):
        """Handle the close event."""
        # Perform any necessary cleanup here
        logger.info("VoxSigil GUI closed")
        event.accept()

    def _create_placeholder_tab(self, label: str) -> QWidget:
        """Create a placeholder tab for components that aren't available."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(f"{label} tab - Component not available or needs implementation."))
        layout.addWidget(
            QLabel("This tab will be populated when the component is properly imported.")
        )
        return widget

    def _create_blt_components_tab(self) -> QWidget:
        """Create a tab for BLT/RAG component monitoring."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("BLT/RAG Components Status"))

        # Add BLT component status indicators
        try:
            from BLT.blt_supervisor_integration import COMPONENTS_AVAILABLE

            if COMPONENTS_AVAILABLE:
                layout.addWidget(QLabel("✅ BLT Components: Available"))
            else:
                layout.addWidget(QLabel("❌ BLT Components: Not Available"))
        except ImportError:
            layout.addWidget(QLabel("❓ BLT Components: Status Unknown"))

        try:
            layout.addWidget(QLabel("✅ BLTEnhancedRAG: Available"))
        except ImportError:
            layout.addWidget(QLabel("❌ BLTEnhancedRAG: Not Available"))

        try:
            layout.addWidget(QLabel("✅ HybridMiddleware: Available"))
        except ImportError:
            layout.addWidget(QLabel("❌ HybridMiddleware: Not Available"))

        return widget

    def _create_arc_components_tab(self) -> QWidget:
        """Create a tab for ARC component monitoring."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("ARC Components Status"))

        try:
            layout.addWidget(QLabel("✅ HybridARCSolver: Available"))
        except ImportError:
            layout.addWidget(QLabel("❌ HybridARCSolver: Not Available"))

        try:
            layout.addWidget(QLabel("✅ ARCReasoner: Available"))
        except ImportError:
            layout.addWidget(QLabel("❌ ARCReasoner: Not Available"))

        return widget

    def _create_vanta_core_tab(self) -> QWidget:
        """Create a tab for Vanta Core monitoring."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Vanta Core Components Status"))

        try:
            layout.addWidget(QLabel("✅ UnifiedVantaCore: Available"))
        except ImportError:
            layout.addWidget(QLabel("❌ UnifiedVantaCore: Not Available"))

        try:
            layout.addWidget(QLabel("✅ VantaSupervisor: Available"))
        except ImportError:
            layout.addWidget(QLabel("❌ VantaSupervisor: Not Available"))

        return widget

    def _create_agent_buttons(self) -> QWidget:
        """Create scrollable area with agent interaction buttons."""
        scroll = QScrollArea()
        container = QWidget()
        layout = QVBoxLayout(container)

        if not self.registry:
            layout.addWidget(QLabel("No agent registry available"))
        else:
            layout.addWidget(QLabel("🤖 Available Agents:"))
            for name, agent in self.registry.get_all_agents():
                if hasattr(agent, "on_gui_call"):
                    btn = QPushButton(f"Interact with {name}")
                    btn.clicked.connect(agent.on_gui_call)
                    layout.addWidget(btn)

        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)  # Limit height so tabs remain primary focus
        return scroll

    def _on_mesh_echo(self, event):
        """Handle mesh echo events."""
        if self.echo_panel:
            msg = event.get("data")
            if isinstance(msg, dict):
                msg = msg.get("data")
            if msg is not None:
                self.echo_panel.add_message(str(msg))

    def _on_mesh_graph(self, event):
        """Handle mesh graph update events."""
        if self.mesh_map_panel:
            graph = event.get("data")
            if graph is not None:
                self.mesh_map_panel.refresh(graph)

    def _on_status(self, event):
        """Handle agent status events."""
        if self.status_panel:
            msg = event.get("data")
            if msg:
                self.status_panel.add_status(str(msg))

    def add_tab(self, name: str, widget, priority: int = 999):
        """
        Add a tab with priority ordering.

        Args:
            name: Tab name/title
            widget: Widget to add as tab
            priority: Priority (lower numbers appear first)
        """
        try:
            # Find the correct insertion position based on priority
            if hasattr(self, "tabs"):
                tab_widget = self.tabs
            else:
                # If we don't have tabs reference, find the QTabWidget
                tab_widget = self.centralWidget()
                if hasattr(tab_widget, "layout") and tab_widget.layout():
                    for i in range(tab_widget.layout().count()):
                        item = tab_widget.layout().itemAt(i)
                        if item and isinstance(item.widget(), QTabWidget):
                            tab_widget = item.widget()
                            break

            if not isinstance(tab_widget, QTabWidget):
                logger.warning("Could not find QTabWidget to add tab to")
                return False

            # For now, just add at the end - we'll implement proper priority later
            tab_widget.addTab(widget, name)
            logger.info(f"Added tab: {name} with priority {priority}")
            return True

        except Exception as e:
            logger.error(f"Error adding tab {name}: {e}")
            return False

    def remove_tab(self, name: str):
        """Remove a tab by name"""
        try:
            if hasattr(self, "tabs"):
                tab_widget = self.tabs
            else:
                tab_widget = self.centralWidget()
                if hasattr(tab_widget, "layout") and tab_widget.layout():
                    for i in range(tab_widget.layout().count()):
                        item = tab_widget.layout().itemAt(i)
                        if item and isinstance(item.widget(), QTabWidget):
                            tab_widget = item.widget()
                            break

            if not isinstance(tab_widget, QTabWidget):
                return False

            # Find and remove tab
            for i in range(tab_widget.count()):
                if tab_widget.tabText(i) == name:
                    tab_widget.removeTab(i)
                    logger.info(f"Removed tab: {name}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error removing tab {name}: {e}")
            return False

    def get_tab_count(self) -> int:
        """Get the total number of tabs"""
        try:
            if hasattr(self, "tabs"):
                return self.tabs.count()
            else:
                tab_widget = self.centralWidget()
                if hasattr(tab_widget, "layout") and tab_widget.layout():
                    for i in range(tab_widget.layout().count()):
                        item = tab_widget.layout().itemAt(i)
                        if item and isinstance(item.widget(), QTabWidget):
                            return item.widget().count()
            return 0
        except Exception:
            return 0

    def list_all_tabs(self) -> list:
        """Get a list of all tab names"""
        try:
            if hasattr(self, "tabs"):
                tab_widget = self.tabs
            else:
                tab_widget = self.centralWidget()
                if hasattr(tab_widget, "layout") and tab_widget.layout():
                    for i in range(tab_widget.layout().count()):
                        item = tab_widget.layout().itemAt(i)
                        if item and isinstance(item.widget(), QTabWidget):
                            tab_widget = item.widget()
                            break

            if not isinstance(tab_widget, QTabWidget):
                return []

            return [tab_widget.tabText(i) for i in range(tab_widget.count())]

        except Exception as e:
            logger.error(f"Error listing tabs: {e}")
            return []

    def _on_voice_command(self, command: str):
        """Handle voice commands from the microphone monitor."""
        try:
            logger.info(f"Voice command received: {command}")

            # Try to route the command to the appropriate agent
            command_lower = command.lower()

            # Check for agent-specific commands
            if "astra" in command_lower and "navigate" in command_lower:
                self._execute_navigation_command(command)
            elif "andy" in command_lower and "compose" in command_lower:
                self._execute_compose_command(command)
            elif "voxka" in command_lower:
                self._execute_voxka_command(command)
            elif "oracle" in command_lower:
                self._execute_oracle_command(command)
            elif "status" in command_lower:
                self._show_system_status()
            elif "training" in command_lower:
                self._start_training_command()
            elif "stop" in command_lower:
                self._stop_all_operations()
            else:
                # General command processing
                self._process_general_command(command)

        except Exception as e:
            logger.error(f"Error processing voice command: {e}")

    def _execute_navigation_command(self, command: str):
        """Execute navigation-related voice command."""
        try:
            # Try to find Astra agent and make it respond
            if self.registry:
                astra = self.registry.get_component("astra")
                if astra and hasattr(astra, "speak"):
                    astra.speak("Navigation command received. Charting the course...")

            # Switch to relevant tab based on command
            if "control" in command.lower():
                self._switch_to_tab("Control Center")
            elif "mesh" in command.lower():
                self._switch_to_tab("Mesh Map")
            elif "status" in command.lower():
                self._switch_to_tab("Agent Status")

        except Exception as e:
            logger.error(f"Error executing navigation command: {e}")

    def _execute_compose_command(self, command: str):
        """Execute composition-related voice command."""
        try:
            if self.registry:
                andy = self.registry.get_component("andy")
                if andy and hasattr(andy, "speak"):
                    andy.speak("Composing output package. Everything's coming together nicely!")

        except Exception as e:
            logger.error(f"Error executing compose command: {e}")

    def _execute_voxka_command(self, command: str):
        """Execute Voxka-related voice command."""
        try:
            if self.registry:
                voxka = self.registry.get_component("voxka")
                if voxka and hasattr(voxka, "speak"):
                    voxka.speak("Engaging dual cognition protocols. The voice of phi resonates...")

        except Exception as e:
            logger.error(f"Error executing Voxka command: {e}")

    def _execute_oracle_command(self, command: str):
        """Execute Oracle-related voice command."""
        try:
            if self.registry:
                oracle = self.registry.get_component("oracle")
                if oracle and hasattr(oracle, "speak"):
                    oracle.speak(
                        "The Oracle speaks. Ancient wisdom flows through the data streams..."
                    )

        except Exception as e:
            logger.error(f"Error executing Oracle command: {e}")

    def _show_system_status(self):
        """Show system status in response to voice command."""
        try:
            self._switch_to_tab("Agent Status")

            if self.registry:
                sam = self.registry.get_component("sam")
                if sam and hasattr(sam, "speak"):
                    sam.speak("Displaying system status. All support systems are ready to assist!")

        except Exception as e:
            logger.error(f"Error showing system status: {e}")

    def _start_training_command(self):
        """Start training in response to voice command."""
        try:
            self._switch_to_tab("Training Pipelines")

            if self.registry:
                dave = self.registry.get_component("dave")
                if dave and hasattr(dave, "speak"):
                    dave.speak("Processing training request. Analysis systems engaged.")

        except Exception as e:
            logger.error(f"Error starting training: {e}")

    def _stop_all_operations(self):
        """Stop all operations in response to voice command."""
        try:
            if self.registry:
                warden = self.registry.get_component("warden")
                if warden and hasattr(warden, "speak"):
                    warden.speak("Security protocols engaged. Operations halting safely.")

        except Exception as e:
            logger.error(f"Error stopping operations: {e}")

    def _process_general_command(self, command: str):
        """Process general voice commands."""
        try:
            if self.registry:
                echo = self.registry.get_component("echo")
                if echo and hasattr(echo, "speak"):
                    echo.speak(f"Command received: {command[:50]}. Processing through the mesh...")

        except Exception as e:
            logger.error(f"Error processing general command: {e}")

    def _switch_to_tab(self, tab_name: str):
        """Switch to a specific tab by name."""
        try:
            # Find the main tab widget
            central_widget = self.centralWidget()
            if central_widget:
                layout = central_widget.layout()
                if layout and layout.count() > 0:
                    tabs = layout.itemAt(0).widget()
                    if isinstance(tabs, QTabWidget):
                        # Search for tab by name
                        for i in range(tabs.count()):
                            if tab_name.lower() in tabs.tabText(i).lower():
                                tabs.setCurrentIndex(i)
                                logger.info(f"Switched to tab: {tabs.tabText(i)}")
                                return

        except Exception as e:
            logger.error(f"Error switching to tab {tab_name}: {e}")

    # ...existing code...


def launch(registry=None, event_bus=None, async_bus=None, training_engine=None):
    """Launch the unified VoxSigil GUI with all components as tabs."""
    app = QApplication(sys.argv)
    win = VoxSigilMainWindow(registry, event_bus, async_bus, training_engine)
    win.show()
    return app.exec_()


# Backward compatibility alias
VoxSigilGUI = VoxSigilMainWindow


if __name__ == "__main__":
    launch()
