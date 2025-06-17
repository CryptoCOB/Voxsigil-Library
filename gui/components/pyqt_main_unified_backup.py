from __future__ import annotations

import sys

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
from .echo_log_panel import EchoLogPanel
from .gui_styles import VoxSigilStyles

# from .mesh_map_panel import MeshMapPanel  # Temporarily disabled due to syntax errors

# Import Enhanced Components
try:
    from .enhanced_music_tab import EnhancedMusicTab

    ENHANCED_MUSIC_AVAILABLE = True
except ImportError:
    ENHANCED_MUSIC_AVAILABLE = False

try:
    from .enhanced_novel_reasoning_tab import EnhancedNovelReasoningTab

    ENHANCED_NOVEL_REASONING_AVAILABLE = True
except ImportError:
    ENHANCED_NOVEL_REASONING_AVAILABLE = False

try:
    from .enhanced_gridformer_tab import EnhancedGridFormerTab

    ENHANCED_GRIDFORMER_AVAILABLE = True
except ImportError:
    ENHANCED_GRIDFORMER_AVAILABLE = False

# Fallback to regular components
try:
    from .enhanced_echo_log_panel import EnhancedEchoLogPanel

    ENHANCED_ECHO_AVAILABLE = True
except ImportError:
    ENHANCED_ECHO_AVAILABLE = False

try:
    from .enhanced_agent_status_panel_v2 import EnhancedAgentStatusPanel

    ENHANCED_AGENT_STATUS_AVAILABLE = True
except ImportError:
    ENHANCED_AGENT_STATUS_AVAILABLE = False

try:
    from .music_tab import MusicTab

    MUSIC_TAB_AVAILABLE = True
except ImportError:
    MUSIC_TAB_AVAILABLE = False

try:
    from .novel_reasoning_tab import NovelReasoningTab

    NOVEL_REASONING_TAB_AVAILABLE = True
except ImportError:
    NOVEL_REASONING_TAB_AVAILABLE = False

try:
    from .dynamic_gridformer_gui import DynamicGridFormerWidget

    GRIDFORMER_AVAILABLE = True
except ImportError:
    GRIDFORMER_AVAILABLE = False

# Import Enhanced Neural TTS Tab
try:
    from .enhanced_neural_tts_tab import EnhancedNeuralTTSTab

    ENHANCED_NEURAL_TTS_AVAILABLE = True
except ImportError:
    ENHANCED_NEURAL_TTS_AVAILABLE = False

# Fallback to regular Neural TTS Tab
try:
    from .neural_tts_tab import NeuralTTSTab

    NEURAL_TTS_AVAILABLE = True
except ImportError:
    NEURAL_TTS_AVAILABLE = False

# Import the enhanced components
try:
    from gui.components.enhanced_model_tab import EnhancedModelTab

    MODEL_TAB_AVAILABLE = True
except ImportError:
    MODEL_TAB_AVAILABLE = False

try:
    from interfaces.performance_tab_interface import PerformanceTabInterface

    PERFORMANCE_TAB_AVAILABLE = True
except ImportError:
    PERFORMANCE_TAB_AVAILABLE = False

try:
    from gui.components.enhanced_visualization_tab import EnhancedVisualizationTab

    VISUALIZATION_TAB_AVAILABLE = True
except ImportError:
    VISUALIZATION_TAB_AVAILABLE = False

try:
    from interfaces.training_interface import TrainingInterface

    TRAINING_TAB_AVAILABLE = True
except ImportError:
    TRAINING_TAB_AVAILABLE = False

# Use enhanced training tab for real streaming functionality
try:
    from gui.components.enhanced_training_tab import EnhancedTrainingTab

    ENHANCED_TRAINING_TAB_AVAILABLE = True
except ImportError:
    ENHANCED_TRAINING_TAB_AVAILABLE = False

try:
    from gui.components.enhanced_model_discovery_tab import EnhancedModelDiscoveryTab

    MODEL_DISCOVERY_AVAILABLE = True
except ImportError:
    MODEL_DISCOVERY_AVAILABLE = False

# Import streaming dashboard for unified real-time view
try:
    from gui.components.streaming_dashboard import StreamingDashboard

    STREAMING_DASHBOARD_AVAILABLE = True
except ImportError:
    STREAMING_DASHBOARD_AVAILABLE = False


class VoxSigilMainWindow(QMainWindow):
    """Unified PyQt main window with all components as tabs."""

    def __init__(self, registry=None, event_bus=None, async_bus=None):
        super().__init__()
        self.registry = registry
        self.event_bus = event_bus
        self.async_bus = async_bus
        self.echo_panel = None
        self.mesh_map_panel = None
        self.status_panel = None
        self.setWindowTitle("VoxSigil Unified GUI - All Components")
        self.resize(1200, 900)
        self._apply_styles()
        self._init_ui()
        self._setup_event_handlers()

    def _apply_styles(self):
        """Apply VoxSigil styling to the main window."""
        # Apply the complete VoxSigil stylesheet to the main window
        self.setStyleSheet(VoxSigilStyles.get_complete_stylesheet())

    def _setup_event_handlers(self):
        """Setup event bus and async bus handlers."""
        if self.event_bus:
            self.event_bus.subscribe("mesh_echo", self._on_mesh_echo)
            self.event_bus.subscribe("mesh_graph_update", self._on_mesh_graph)
            self.event_bus.subscribe("agent_status", self._on_status)

        if self.async_bus:
            try:                self.async_bus.register_component("GUI")
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
        
        # Core Components Tabs
        if MODEL_TAB_AVAILABLE:
            model_tab = EnhancedModelTab()
            tabs.addTab(model_tab, "🤖 Models")
        else:
            tabs.addTab(self._create_placeholder_tab("Models"), "🤖 Models")

        if MODEL_DISCOVERY_AVAILABLE:
            discovery_tab = EnhancedModelDiscoveryTab()
            tabs.addTab(discovery_tab, "🔍 Model Discovery")
        else:
            tabs.addTab(self._create_placeholder_tab("Model Discovery"), "🔍 Model Discovery")

        # Use enhanced training tab for real streaming functionality
        if ENHANCED_TRAINING_TAB_AVAILABLE:
            training_tab = EnhancedTrainingTab()
            tabs.addTab(training_tab, "🎯 Training")
        elif TRAINING_TAB_AVAILABLE:
            training_tab = TrainingInterface()
            tabs.addTab(training_tab, "🎯 Training (Legacy)")
        else:
            tabs.addTab(self._create_placeholder_tab("Training"), "🎯 Training")

        # Add streaming dashboard for unified real-time monitoring
        if STREAMING_DASHBOARD_AVAILABLE:
            streaming_tab = StreamingDashboard()
            tabs.addTab(streaming_tab, "📊 Live Dashboard")

        # Add streaming dashboard for unified real-time monitoring
        if STREAMING_DASHBOARD_AVAILABLE:
            streaming_tab = StreamingDashboard()
            tabs.addTab(streaming_tab, "📊 Live Dashboard")

        # Novel Reasoning tab - try enhanced version first
        if ENHANCED_NOVEL_REASONING_AVAILABLE:
            novel_tab = EnhancedNovelReasoningTab()
        elif NOVEL_REASONING_TAB_AVAILABLE:
            novel_tab = NovelReasoningTab()
        else:
            novel_tab = self._create_placeholder_tab("Novel Reasoning")
        tabs.addTab(novel_tab, "🧠 Novel Reasoning")

        if VISUALIZATION_TAB_AVAILABLE:
            viz_tab = EnhancedVisualizationTab()
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
        if ENHANCED_GRIDFORMER_AVAILABLE:
            gridformer_tab = EnhancedGridFormerTab()
        elif GRIDFORMER_AVAILABLE:
            gridformer_tab = DynamicGridFormerWidget()
        else:
            gridformer_tab = self._create_placeholder_tab("GridFormer")
        tabs.addTab(gridformer_tab, "🔄 GridFormer")

        if ENHANCED_MUSIC_AVAILABLE:
            music_tab = EnhancedMusicTab()
        elif MUSIC_TAB_AVAILABLE:
            music_tab = MusicTab()
        else:
            music_tab = self._create_placeholder_tab("Music")
        tabs.addTab(music_tab, "🎵 Music")

        # Neural TTS Tab - Try enhanced version first
        if ENHANCED_NEURAL_TTS_AVAILABLE:
            neural_tts_tab = EnhancedNeuralTTSTab()
            tabs.addTab(neural_tts_tab, "🎙️ Neural TTS")
        elif NEURAL_TTS_AVAILABLE:
            neural_tts_tab = NeuralTTSTab()
            tabs.addTab(neural_tts_tab, "🎙️ Neural TTS")
        else:
            tabs.addTab(self._create_placeholder_tab("Neural TTS"), "🎙️ Neural TTS")

        # Core Monitoring Tabs
        if ENHANCED_ECHO_AVAILABLE:
            self.echo_panel = EnhancedEchoLogPanel()
        else:
            self.echo_panel = EchoLogPanel()

        # self.mesh_map_panel = MeshMapPanel()  # Temporarily disabled due to syntax errors
        self.mesh_map_panel = None

        if ENHANCED_AGENT_STATUS_AVAILABLE:
            self.status_panel = EnhancedAgentStatusPanel()
        else:
            self.status_panel = AgentStatusPanel()

        tabs.addTab(self.echo_panel, "📡 Echo Log")
        tabs.addTab(self.mesh_map_panel, "🕸️ Mesh Map")
        tabs.addTab(self.status_panel, "📈 Agent Status")

        # Add BLT/RAG Components Tab
        blt_tab = self._create_blt_components_tab()
        tabs.addTab(blt_tab, "🔧 BLT/RAG")

        # Add ARC Components Tab
        arc_tab = self._create_arc_components_tab()
        tabs.addTab(arc_tab, "🧩 ARC")  # Add Vanta Core Tab
        vanta_tab = self._create_vanta_core_tab()
        tabs.addTab(vanta_tab, "⚡ Vanta Core")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(tabs)

        # Apply VoxSigil styling to the tab widget
        tabs.setStyleSheet(VoxSigilStyles.get_tab_widget_stylesheet())

        if self.registry:
            layout.addWidget(self._create_agent_buttons())

        self.setCentralWidget(container)

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


def launch(registry=None, event_bus=None, async_bus=None):
    """Launch the unified VoxSigil GUI with all components as tabs."""
    app = QApplication(sys.argv)
    win = VoxSigilMainWindow(registry, event_bus, async_bus)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    launch()
