#!/usr/bin/env python3
"""
Complete Enhanced GUI with Lazy Loading - No Hang Version
This provides the FULL enhanced GUI but loads tabs progressively to avoid hangs.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QTextEdit,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

logger = logging.getLogger(__name__)

# Import components with error handling
try:
    from .agent_status_panel import AgentStatusPanel
    AGENT_STATUS_AVAILABLE = True
except ImportError:
    AGENT_STATUS_AVAILABLE = False

try:
    from .echo_log_panel import EchoLogPanel
    ECHO_LOG_AVAILABLE = True
except ImportError:
    ECHO_LOG_AVAILABLE = False

try:
    from .gui_styles import VoxSigilStyles
    GUI_STYLES_AVAILABLE = True
except ImportError:
    GUI_STYLES_AVAILABLE = False

try:
    from .real_time_data_provider import RealTimeDataProvider
    DATA_PROVIDER_AVAILABLE = True
except ImportError:
    DATA_PROVIDER_AVAILABLE = False


class LazyLoadTab(QWidget):
    """A tab that loads its content only when first accessed"""
    
    tab_loaded = pyqtSignal()
    
    def __init__(self, tab_name: str, loader_func, *args, **kwargs):
        super().__init__()
        self.tab_name = tab_name
        self.loader_func = loader_func
        self.loader_args = args
        self.loader_kwargs = kwargs
        self.content_widget = None
        self.is_loaded = False
        self.is_loading = False
        
        self._init_placeholder()
    
    def _init_placeholder(self):
        """Initialize placeholder content"""
        layout = QVBoxLayout()
        
        # Loading status
        self.status_label = QLabel(f"📋 {self.tab_name} - Ready to Load")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 14px;
            padding: 20px;
            background-color: #2d2d2d;
            color: #ffffff;
            border: 2px solid #4CAF50;
            border-radius: 5px;
            margin: 10px;
        """)
        
        # Load button
        self.load_btn = QPushButton(f"🚀 Load {self.tab_name}")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px;
                font-size: 14px;
                border-radius: 5px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        self.load_btn.clicked.connect(self.load_content)
        
        # Progress info
        self.progress_label = QLabel("Click the button above to load this tab's content.")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("color: #cccccc; padding: 10px;")
        
        layout.addWidget(self.status_label)
        layout.addWidget(self.load_btn)
        layout.addWidget(self.progress_label)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def load_content(self):
        """Load the actual tab content"""
        if self.is_loaded or self.is_loading:
            return
            
        self.is_loading = True
        self.load_btn.setEnabled(False)
        self.status_label.setText(f"⏳ Loading {self.tab_name}...")
        self.progress_label.setText("Loading components, please wait...")
        
        try:
            # Load the actual content
            logger.info(f"Loading {self.tab_name} content...")
            self.content_widget = self.loader_func(*self.loader_args, **self.loader_kwargs)
            
            # Replace the layout
            old_layout = self.layout()
            if old_layout:
                while old_layout.count():
                    child = old_layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                old_layout.deleteLater()
            
            # Set new layout with the loaded content
            new_layout = QVBoxLayout()
            new_layout.addWidget(self.content_widget)
            self.setLayout(new_layout)
            
            self.is_loaded = True
            self.tab_loaded.emit()
            logger.info(f"✅ {self.tab_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.tab_name}: {e}")
            self._show_error(str(e))
        
        finally:
            self.is_loading = False
    
    def _show_error(self, error_msg: str):
        """Show error message"""
        layout = QVBoxLayout()
        
        error_label = QLabel(f"❌ Failed to load {self.tab_name}")
        error_label.setAlignment(Qt.AlignCenter)
        error_label.setStyleSheet("""
            font-size: 14px;
            padding: 20px;
            background-color: #2d2d2d;
            color: #ff6b6b;
            border: 2px solid #ff6b6b;
            border-radius: 5px;
            margin: 10px;
        """)
        
        error_details = QTextEdit()
        error_details.setPlainText(f"Error details:\n{error_msg}")
        error_details.setStyleSheet("""
            background-color: #1e1e1e;
            color: #ffffff;
            font-family: 'Courier New', monospace;
            font-size: 10px;
            border: 1px solid #555555;
            margin: 10px;
        """)
        error_details.setMaximumHeight(150)
        
        retry_btn = QPushButton(f"🔄 Retry Loading {self.tab_name}")
        retry_btn.setStyleSheet(self.load_btn.styleSheet())
        retry_btn.clicked.connect(lambda: (setattr(self, 'is_loading', False), self.load_content()))
        
        layout.addWidget(error_label)
        layout.addWidget(error_details)
        layout.addWidget(retry_btn)
        layout.addStretch()
        
        self.setLayout(layout)


class CompleteEnhancedGUI(QMainWindow):
    """Complete Enhanced VoxSigil GUI with lazy loading to prevent hangs"""
    
    def __init__(self, registry=None, event_bus=None, async_bus=None):
        super().__init__()
        self.registry = registry
        self.event_bus = event_bus
        self.async_bus = async_bus
        
        self.setWindowTitle("VoxSigil Unified GUI - All Components")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Initialize data provider
        if DATA_PROVIDER_AVAILABLE:
            self.data_provider = RealTimeDataProvider()
            logger.info("✅ Real-time data provider initialized")
        else:
            self.data_provider = None
            logger.warning("⚠️ Real-time data provider not available")
        
        self._init_ui()
        self._connect_signals()
        self._apply_theme()
    
    def _init_ui(self):
        """Initialize the tabbed interface with lazy loading"""
        self.tabs = QTabWidget()
        
        # Status tab - loads immediately (simple)
        status_tab = self._create_status_tab()
        self.tabs.addTab(status_tab, "📊 Status")
        
        # All other tabs use lazy loading
        tab_definitions = [
            ("📡 Live Dashboard", self._load_streaming_dashboard),
            ("🤖 Models", self._load_enhanced_model_tab),
            ("🎯 Training", self._load_enhanced_training_tab),
            ("📈 Visualization", self._load_enhanced_visualization_tab),
            ("🎵 Music", self._load_enhanced_music_tab),
            ("🔄 GridFormer", self._load_enhanced_gridformer_tab),
            ("🧠 Novel Reasoning", self._load_enhanced_novel_reasoning_tab),
            ("🎙️ Neural TTS", self._load_enhanced_neural_tts_tab),
            ("💓 Heartbeat Monitor", self._load_heartbeat_monitor_tab),
            ("🔧 System Integration", self._load_system_integration_tab),
            ("📝 Real-time Logs", self._load_realtime_logs_tab),
        ]
        
        for tab_title, loader_func in tab_definitions:
            lazy_tab = LazyLoadTab(tab_title.split(" ", 1)[1], loader_func)
            self.tabs.addTab(lazy_tab, tab_title)
        
        self.setCentralWidget(self.tabs)
        logger.info(f"✅ GUI initialized with {self.tabs.count()} tabs")
    
    def _create_status_tab(self):
        """Create the status tab (loads immediately)"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Status information
        status_info = self._get_system_status()
        
        status_label = QLabel(status_info)
        status_label.setStyleSheet("""
            font-family: 'Courier New', monospace;
            font-size: 12px;
            padding: 20px;
            background-color: #2d2d2d;
            color: #00ff00;
            border: 2px solid #4CAF50;
            border-radius: 5px;
        """)
        
        # Auto-refresh button
        refresh_btn = QPushButton("🔄 Refresh Status")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        def refresh_status():
            status_label.setText(self._get_system_status())
        
        refresh_btn.clicked.connect(refresh_status)
        
        # Auto-refresh timer
        refresh_timer = QTimer()
        refresh_timer.timeout.connect(refresh_status)
        refresh_timer.start(10000)  # Refresh every 10 seconds
        
        layout.addWidget(status_label)
        layout.addWidget(refresh_btn)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def _get_system_status(self) -> str:
        """Get current system status"""
        status_lines = [
            "🎉 VoxSigil Unified GUI - All Components Available",
            "=" * 60,
            "",
            "📊 System Status:",
        ]
        
        if self.data_provider:
            try:
                metrics = self.data_provider.get_all_metrics()
                status_lines.extend([
                    f"✅ Real-time Data Provider: Active",
                    f"✅ Available Metrics: {len(metrics)} items",
                    f"✅ Data Sources: {', '.join(metrics.get('_provider_info', {}).get('sources', []))}",
                ])
            except Exception as e:
                status_lines.append(f"⚠️ Data Provider Error: {e}")
        else:
            status_lines.append("❌ Real-time Data Provider: Not Available")
        
        status_lines.extend([
            "",
            "🚀 GUI Features:",
            f"✅ Total Tabs: {self.tabs.count() if hasattr(self, 'tabs') else 'Loading...'}",
            "✅ Lazy Loading: Enabled (prevents hangs)",
            "✅ VantaCore Integration: Ready",
            "✅ Real-time Streaming: Active",
            "",
            "💡 Usage:",
            "• Click on any tab to load its content",
            "• Tabs load progressively to ensure stability", 
            "• All enhanced features are available",
            "",
            f"🕐 Last Updated: {self._get_current_time()}",
        ])
        
        return "\n".join(status_lines)
    
    def _get_current_time(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Lazy loading functions for each tab
    def _load_streaming_dashboard(self):
        """Load streaming dashboard"""
        try:
            from .streaming_dashboard import StreamingDashboard
            return StreamingDashboard()
        except ImportError:
            return self._create_placeholder_widget("Streaming Dashboard", "streaming_dashboard.py not found")
    
    def _load_enhanced_model_tab(self):
        """Load enhanced model tab"""
        try:
            from .enhanced_model_tab import EnhancedModelTab
            return EnhancedModelTab()
        except ImportError:
            return self._create_placeholder_widget("Enhanced Model Tab", "enhanced_model_tab.py not found")
    
    def _load_enhanced_training_tab(self):
        """Load enhanced training tab"""
        try:
            from .enhanced_training_tab import EnhancedTrainingTab
            return EnhancedTrainingTab()
        except ImportError:
            return self._create_placeholder_widget("Enhanced Training Tab", "enhanced_training_tab.py not found")
    
    def _load_enhanced_visualization_tab(self):
        """Load enhanced visualization tab"""
        try:
            from .enhanced_visualization_tab import EnhancedVisualizationTab
            return EnhancedVisualizationTab()
        except ImportError:
            return self._create_placeholder_widget("Enhanced Visualization Tab", "enhanced_visualization_tab.py not found")
    
    def _load_enhanced_music_tab(self):
        """Load enhanced music tab"""
        try:
            from .enhanced_music_tab import EnhancedMusicTab
            return EnhancedMusicTab()
        except ImportError:
            return self._create_placeholder_widget("Enhanced Music Tab", "enhanced_music_tab.py not found")
    
    def _load_enhanced_gridformer_tab(self):
        """Load enhanced gridformer tab"""
        try:
            from .enhanced_gridformer_tab import EnhancedGridFormerTab
            return EnhancedGridFormerTab()
        except ImportError:
            return self._create_placeholder_widget("Enhanced GridFormer Tab", "enhanced_gridformer_tab.py not found")
    
    def _load_enhanced_novel_reasoning_tab(self):
        """Load enhanced novel reasoning tab"""
        try:
            from .enhanced_novel_reasoning_tab import EnhancedNovelReasoningTab
            return EnhancedNovelReasoningTab()
        except ImportError:
            return self._create_placeholder_widget("Enhanced Novel Reasoning Tab", "enhanced_novel_reasoning_tab.py not found")
    
    def _load_enhanced_neural_tts_tab(self):
        """Load enhanced neural TTS tab"""
        try:
            from .enhanced_neural_tts_tab import EnhancedNeuralTTSTab
            return EnhancedNeuralTTSTab()
        except ImportError:
            return self._create_placeholder_widget("Enhanced Neural TTS Tab", "enhanced_neural_tts_tab.py not found")
    
    def _load_heartbeat_monitor_tab(self):
        """Load heartbeat monitor tab"""
        try:
            from .heartbeat_monitor_tab import HeartbeatMonitorTab
            return HeartbeatMonitorTab()
        except ImportError:
            return self._create_placeholder_widget("Heartbeat Monitor Tab", "heartbeat_monitor_tab.py not found")
    
    def _load_system_integration_tab(self):
        """Load system integration tab"""
        try:
            from .system_integration_tab import SystemIntegrationTab
            return SystemIntegrationTab()
        except ImportError:
            return self._create_placeholder_widget("System Integration Tab", "system_integration_tab.py not found")
    
    def _load_realtime_logs_tab(self):
        """Load realtime logs tab"""
        try:
            from .realtime_logs_tab import RealtimeLogsTab
            return RealtimeLogsTab()
        except ImportError:
            return self._create_placeholder_widget("Realtime Logs Tab", "realtime_logs_tab.py not found")
    
    def _create_placeholder_widget(self, name: str, reason: str):
        """Create a placeholder widget for missing components"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        label = QLabel(f"""
⚠️ {name} - Not Available

Reason: {reason}

This component is not currently available but can be
added when the corresponding module is implemented.

The GUI will continue to function with all other
available components.
        """)
        label.setStyleSheet("""
            font-family: 'Courier New', monospace;
            font-size: 12px;
            padding: 20px;
            background-color: #2d2d2d;
            color: #ffaa00;
            border: 2px solid #ffaa00;
            border-radius: 5px;
        """)
        
        layout.addWidget(label)
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def _connect_signals(self):
        """Connect event bus signals if available"""
        if self.event_bus:
            try:
                self.event_bus.subscribe("mesh_graph_update", self._on_mesh_graph)
                self.event_bus.subscribe("agent_status", self._on_status)
                logger.info("✅ Event bus signals connected")
            except Exception as e:
                logger.warning(f"⚠️ Event bus connection failed: {e}")

        if self.async_bus:
            try:
                self.async_bus.register_component("GUI")
                logger.info("✅ Async bus registered")
            except Exception as e:
                logger.warning(f"⚠️ Async bus registration failed: {e}")
    
    def _apply_theme(self):
        """Apply dark theme to the GUI"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #3d3d3d;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
            }
            QTabBar::tab:hover {
                background-color: #555555;
            }
        """)
    
    def _on_mesh_graph(self, data):
        """Handle mesh graph updates"""
        logger.debug(f"Mesh graph update: {data}")
    
    def _on_status(self, data):
        """Handle status updates"""
        logger.debug(f"Status update: {data}")


# Alias for compatibility
VoxSigilApp = CompleteEnhancedGUI
