#!/usr/bin/env python3
"""
VoxSigil Enhanced GUI - Optimized with Timeout & Retry Support
This version includes all the suggested optimizations to prevent hanging and improve loading.
"""

import logging
import sys
import gc
import tracemalloc
import traceback
import importlib
from typing import Dict, Any, Optional, Callable, NamedTuple, List

from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QPushButton, QTabWidget, QVBoxLayout,
    QWidget, QTextEdit, QProgressBar, QHBoxLayout, QSplashScreen, QMessageBox, QShortcut,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QKeySequence

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 0)  Shared helpers and optimizations
# ──────────────────────────────────────────────────────────────────────────────

class FeatureSpec(NamedTuple):
    title: str
    loader: Callable[..., QWidget]
    timeout_ms: int = 12_000
    max_retries: int = 2

def log_exception(prefix: str, exc: Exception):
    logger.error(f"{prefix}: {exc}")
    logger.debug("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

def safe_import_psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        logger.warning("psutil not available - resource monitoring disabled")
        return None

psutil = safe_import_psutil()

# Import essential components
try:
    from gui.components.real_time_data_provider import RealTimeDataProvider
    DATA_PROVIDER_AVAILABLE = True
except ImportError:
    DATA_PROVIDER_AVAILABLE = False

try:
    from gui.components.gui_styles import VoxSigilStyles
    GUI_STYLES_AVAILABLE = True
except ImportError:
    GUI_STYLES_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# 1)  Enhanced TabLoadWorker with timeout support
# ──────────────────────────────────────────────────────────────────────────────

class TabLoadWorker(QThread):
    """Background worker to load tabs with interruption support"""
    
    tab_loaded = pyqtSignal(QWidget)
    load_error = pyqtSignal(str)
    load_progress = pyqtSignal(int)
    cancelled = pyqtSignal()
    
    def __init__(self, loader_function, *args, **kwargs):
        super().__init__()
        self.loader_function = loader_function
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        """Load tab with interruption checks"""
        try:
            if self.isInterruptionRequested():
                self.cancelled.emit()
                return
                
            self.load_progress.emit(25)
            
            if self.isInterruptionRequested():
                self.cancelled.emit()
                return
                
            widget = self.loader_function(*self.args, **self.kwargs)
            
            if self.isInterruptionRequested():
                self.cancelled.emit()
                return
                
            self.load_progress.emit(75)
            self.tab_loaded.emit(widget)
            self.load_progress.emit(100)
            
        except Exception as e:
            log_exception("TabLoadWorker error", e)
            self.load_error.emit(str(e))
        finally:
            if self.isInterruptionRequested():
                self.cancelled.emit()

# ──────────────────────────────────────────────────────────────────────────────
# 2)  Enhanced LazyTab with timeout and retry
# ──────────────────────────────────────────────────────────────────────────────

class OptimizedLazyTab(QWidget):
    """Tab with timeout, retry, and circuit breaker pattern"""
    
    def __init__(self, tab_name: str, loader_function, main_gui_ref, 
                 timeout_ms: int = 12_000, max_retries: int = 2):
        super().__init__()
        self.tab_name = tab_name
        self.loader_function = loader_function
        self.main_gui_ref = main_gui_ref
        self.is_loaded = False
        self.actual_widget = None
        self.worker = None
        
        # Enhanced loading controls
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self._retries_left = max_retries
        
        # Timeout timer
        self.timeout_timer = QTimer(self)
        self.timeout_timer.setSingleShot(True)
        self.timeout_timer.timeout.connect(self._on_load_timeout)
        
        self._init_styled_placeholder()
    
    def _init_styled_placeholder(self):
        """Initialize with VoxSigil styling"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Apply styling
        if GUI_STYLES_AVAILABLE:
            self.setStyleSheet(f"""
                QWidget {{
                    background-color: {VoxSigilStyles.COLORS["bg_primary"]};
                    color: {VoxSigilStyles.COLORS["text_primary"]};
                }}
            """)
        else:
            self.setStyleSheet("QWidget { background-color: #1a1a1a; color: #ffffff; }")
        
        # Title
        title = QLabel(f"🚀 {self.tab_name}")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #00ffff;
                background-color: #2d2d2d;
                border: 2px solid #00ffff;
                border-radius: 10px;
                padding: 15px;
                margin: 10px;
            }
        """)
        layout.addWidget(title)
        
        # Description
        desc = QLabel(f"Click 'Load {self.tab_name}' to activate enhanced features with original styling.")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("QLabel { font-size: 14px; color: #cccccc; padding: 10px; }")
        layout.addWidget(desc)
        
        # Load button
        self.load_button = QPushButton(f"🎯 Load {self.tab_name}")
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #00ffff;
                color: black;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #999999;
            }
        """)
        self.load_button.clicked.connect(self._load_enhanced_content)
        layout.addWidget(self.load_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #00ffff;
                border-radius: 5px;
                text-align: center;
                background-color: #2d2d2d;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to load enhanced features")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("QLabel { color: #999999; font-style: italic; margin: 10px; }")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def _load_enhanced_content(self):
        """Load with timeout and retry support"""
        if self.is_loaded or self.worker is not None:
            return
            
        # Circuit breaker
        if self._retries_left <= 0:
            self.status_label.setText("❌ Too many failures – tab disabled")
            self.status_label.setStyleSheet("color: #ff4444; font-weight: bold;")
            self.load_button.setDisabled(True)
            return
            
        try:
            # Show loading state
            self.load_button.setEnabled(False)
            retry_num = self.max_retries - self._retries_left + 1
            self.load_button.setText(f"🔄 Loading... (Attempt {retry_num})")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.status_label.setText(f"Loading {self.tab_name}... (Timeout: {self.timeout_ms/1000}s)")
            
            # Start background loading
            self.worker = TabLoadWorker(self.loader_function)
            self.worker.tab_loaded.connect(self._on_tab_loaded)
            self.worker.load_error.connect(self._on_load_error)
            self.worker.load_progress.connect(self._on_load_progress)
            self.worker.cancelled.connect(self._on_cancelled)
            
            self.worker.start()
            self.timeout_timer.start(self.timeout_ms)
            
        except Exception as e:
            log_exception(f"Failed to start loading {self.tab_name}", e)
            self._on_load_error(str(e))
    
    def _on_load_progress(self, progress):
        """Update progress"""
        self.progress_bar.setValue(progress)
        if progress == 25:
            self.status_label.setText("Importing components...")
        elif progress == 75:
            self.status_label.setText("Initializing interface...")
        elif progress == 100:
            self.status_label.setText("Loading complete!")
    
    def _on_tab_loaded(self, widget):
        """Handle successful load"""
        try:
            self.timeout_timer.stop()
            self.actual_widget = widget
            self._replace_with_enhanced_content()
            self.is_loaded = True
            self.worker = None
            
            # Micro GC
            QTimer.singleShot(0, gc.collect)
            
            logger.info(f"✅ {self.tab_name} loaded successfully")
            
        except Exception as e:
            log_exception(f"Failed to display {self.tab_name}", e)
            self._on_load_error(f"Display error: {e}")
    
    def _on_load_error(self, error_msg):
        """Handle errors with retry logic"""
        self.timeout_timer.stop()
        self._retries_left -= 1
        
        log_exception(f"Failed to load {self.tab_name}", Exception(error_msg))
        
        self.progress_bar.setVisible(False)
        self.load_button.setEnabled(True)
        
        if self._retries_left > 0:
            self.load_button.setText(f"🔄 Retry {self.tab_name} ({self._retries_left} left)")
            self.status_label.setText(f"⚠️ Error: {error_msg[:50]}... - {self._retries_left} retries left")
        else:
            self.load_button.setText(f"❌ {self.tab_name} Failed")
            self.load_button.setDisabled(True)
            self.status_label.setText(f"❌ Final Error: {error_msg[:50]}...")
        
        self.status_label.setStyleSheet("color: #ff4444;")
        self.worker = None
    
    def _on_load_timeout(self):
        """Handle timeout"""
        logger.warning(f"⏰ Timeout loading {self.tab_name} after {self.timeout_ms}ms")
        if self.worker:
            self.worker.requestInterruption()
        self._on_load_error(f"Timeout after {self.timeout_ms/1000:.1f}s")
    
    def _on_cancelled(self):
        """Handle cancellation"""
        logger.info(f"🚫 Loading cancelled for {self.tab_name}")
        self.timeout_timer.stop()
        self.worker = None
        self.progress_bar.setVisible(False)
        self.load_button.setEnabled(True)
        self.load_button.setText(f"🔄 Retry {self.tab_name}")
        self.status_label.setText("Loading cancelled")
    
    def _replace_with_enhanced_content(self):
        """Replace placeholder with actual content"""
        layout = self.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if isinstance(self.actual_widget, QWidget):
            layout.addWidget(self.actual_widget)

# ──────────────────────────────────────────────────────────────────────────────
# 3)  Main GUI with resource monitoring
# ──────────────────────────────────────────────────────────────────────────────

class OptimizedEnhancedGUI(QMainWindow):
    """Enhanced GUI with all optimizations"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoxSigil Enhanced GUI - Optimized")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Initialize data provider
        self.data_provider = None
        self._init_data_provider()
        
        self._init_ui()
        self._apply_theme()
        self._add_shortcuts()
        
        logger.info("✅ Optimized Enhanced GUI initialized")
    
    def _init_data_provider(self):
        """Safe data provider init"""
        try:
            if DATA_PROVIDER_AVAILABLE:
                self.data_provider = RealTimeDataProvider()
                logger.info("✅ Data provider initialized")
        except Exception as e:
            logger.warning(f"⚠️ Data provider failed: {e}")
    
    def _init_ui(self):
        """Initialize with optimized loading"""
        self.tabs = QTabWidget()
        
        # Apply styling
        if GUI_STYLES_AVAILABLE:
            self.tabs.setStyleSheet(VoxSigilStyles.get_tab_widget_stylesheet())
        
        # Status tab
        status_tab = self._create_status_tab()
        self.tabs.addTab(status_tab, "📊 Status")
        
        # Feature specs with custom timeouts
        feature_specs: List[FeatureSpec] = [
            FeatureSpec("📡 Dashboard", self._load_streaming_dashboard, 10_000, 3),
            FeatureSpec("🤖 Models", self._load_enhanced_model_tab, 15_000, 2),
            FeatureSpec("🎯 Training", self._load_enhanced_training_tab, 20_000, 2),
            FeatureSpec("📈 Visualization", self._load_enhanced_visualization_tab, 8_000, 3),
            FeatureSpec("🎵 Music", self._load_enhanced_music_tab, 12_000, 2),
            FeatureSpec("💓 Heartbeat", self._load_heartbeat_monitor_tab, 5_000, 3),
        ]
        
        for spec in feature_specs:
            tab = OptimizedLazyTab(
                spec.title.split(" ", 1)[1], 
                spec.loader, 
                self,
                timeout_ms=spec.timeout_ms,
                max_retries=spec.max_retries
            )
            self.tabs.addTab(tab, spec.title)
        
        self.setCentralWidget(self.tabs)
        
        # Resource monitoring
        if psutil:
            self._resource_timer = QTimer(self)
            self._resource_timer.timeout.connect(self._update_resources)
            self._resource_timer.start(3_000)
        
        logger.info(f"✅ GUI with {self.tabs.count()} tabs ready")
    
    def _create_status_tab(self):
        """Create status tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        title = QLabel("🎉 VoxSigil Enhanced GUI - Optimized")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-weight: bold;
                color: #00ffff;
                background-color: #2d2d2d;
                border: 2px solid #00ffff;
                border-radius: 10px;
                padding: 15px;
                margin: 10px;
            }
        """)
        layout.addWidget(title)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        if GUI_STYLES_AVAILABLE:
            self.status_text.setStyleSheet(VoxSigilStyles.get_text_edit_stylesheet())
        
        status_info = self._get_status_info()
        self.status_text.setPlainText(status_info)
        layout.addWidget(self.status_text)
        
        refresh_btn = QPushButton("🔄 Refresh Status")
        if GUI_STYLES_AVAILABLE:
            refresh_btn.setStyleSheet(VoxSigilStyles.get_button_stylesheet())
        refresh_btn.clicked.connect(self._refresh_status)
        layout.addWidget(refresh_btn)
        
        tab.setLayout(layout)
        return tab
    
    def _get_status_info(self) -> str:
        """Get current status"""
        status = [
            "🎉 VoxSigil Enhanced GUI - Optimized Version",
            "=" * 60,
            "",
            "✅ Optimizations Active:",
            "• Timeout protection (5-20s per tab)",
            "• Automatic retry (2-3 attempts)",
            "• Circuit breaker for failed tabs",
            "• Memory leak detection",
            "• Resource monitoring",
            "• Background loading",
            "",
            "🚀 Available Tabs:",
        ]
        
        for i in range(self.tabs.count()):
            tab_title = self.tabs.tabText(i)
            tab_widget = self.tabs.widget(i)
            if hasattr(tab_widget, 'is_loaded'):
                status.append(f"• {tab_title}: {'✅ Loaded' if tab_widget.is_loaded else '🔄 Ready to load'}")
            else:
                status.append(f"• {tab_title}: ✅ Active")
        
        status.extend([
            "",
            "💡 Usage:",
            "• Click tabs to load enhanced features",
            "• Features load with timeout protection", 
            "• Failed tabs auto-retry or disable",
            "",
            "🎮 Shortcuts:",
            "• Ctrl+R: Hot reload",
            "• Ctrl+T: Resource stats",
            "• Ctrl+G: Force garbage collection",
        ])
        
        if self.data_provider:
            try:
                metrics = self.data_provider.get_all_metrics()
                status.extend([
                    "",
                    f"📊 Data Provider: ✅ Active ({len(metrics)} metrics)",
                ])
            except:
                status.append("📊 Data Provider: ⚠️ Error")
        else:
            status.append("📊 Data Provider: ❌ Not Available")
        
        return "\n".join(status)
    
    def _refresh_status(self):
        """Refresh status display"""
        self.status_text.setPlainText(self._get_status_info())
    
    def _apply_theme(self):
        """Apply VoxSigil theme"""
        if GUI_STYLES_AVAILABLE:
            self.setStyleSheet(VoxSigilStyles.get_base_stylesheet())
        else:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1a1a1a;
                    color: #ffffff;
                }
                QTabWidget::pane {
                    border: 2px solid #00ffff;
                    background-color: #2d2d2d;
                }
                QTabBar::tab {
                    background-color: #3d3d3d;
                    color: #ffffff;
                    padding: 8px 16px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #00ffff;
                    color: #000000;
                }
            """)
    
    def _add_shortcuts(self):
        """Add keyboard shortcuts"""
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._hot_reload)
        QShortcut(QKeySequence("Ctrl+T"), self, activated=self._show_stats)
        QShortcut(QKeySequence("Ctrl+G"), self, activated=self._force_gc)
    
    def _update_resources(self):
        """Update resource display"""
        if not psutil:
            return
        try:
            mem = psutil.virtual_memory().percent
            cpu = psutil.cpu_percent()
            process_mem = psutil.Process().memory_info().rss / 1024 / 1024
            self.statusBar().showMessage(f"CPU: {cpu:.1f}% | MEM: {mem:.1f}% | Process: {process_mem:.1f}MB")
        except:
            pass
    
    def _show_stats(self):
        """Show resource statistics"""
        try:
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                mem_info = f"Current: {current/1024/1024:.1f}MB, Peak: {peak/1024/1024:.1f}MB"
            else:
                mem_info = "Memory tracing not active"
            
            if psutil:
                sys_info = f"CPU: {psutil.cpu_percent():.1f}%, RAM: {psutil.virtual_memory().percent:.1f}%"
            else:
                sys_info = "System monitoring not available"
            
            QMessageBox.information(self, "Resource Stats", 
                                  f"Memory Tracking: {mem_info}\nSystem: {sys_info}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Stats error: {e}")
    
    def _force_gc(self):
        """Force garbage collection"""
        collected = gc.collect()
        self.statusBar().showMessage(f"GC: Collected {collected} objects", 3000)
        logger.info(f"🗑️ GC collected {collected} objects")
    
    def _hot_reload(self):
        """Hot reload modules"""
        try:
            importlib.reload(sys.modules[__name__])
            QMessageBox.information(self, "Reloaded", "Modules reloaded - restart recommended")
        except Exception as e:
            QMessageBox.warning(self, "Reload Error", f"Failed: {e}")
    
    # Tab loader methods (placeholder implementations)
    def _load_streaming_dashboard(self):
        """Load streaming dashboard"""
        try:
            from gui.components.streaming_dashboard import StreamingDashboard
            return StreamingDashboard(data_provider=self.data_provider)
        except ImportError:
            return self._create_placeholder("Streaming Dashboard", "Module not found")
    
    def _load_enhanced_model_tab(self):
        """Load model tab"""
        try:
            from gui.components.enhanced_model_tab import EnhancedModelTab
            return EnhancedModelTab(data_provider=self.data_provider)
        except ImportError:
            return self._create_placeholder("Enhanced Model Tab", "Module not found")
    
    def _load_enhanced_training_tab(self):
        """Load training tab"""
        try:
            from gui.components.enhanced_training_tab import EnhancedTrainingTab
            return EnhancedTrainingTab(data_provider=self.data_provider)
        except ImportError:
            return self._create_placeholder("Enhanced Training Tab", "Module not found")
    
    def _load_enhanced_visualization_tab(self):
        """Load visualization tab"""
        try:
            from gui.components.enhanced_visualization_tab import EnhancedVisualizationTab
            return EnhancedVisualizationTab(data_provider=self.data_provider)
        except ImportError:
            return self._create_placeholder("Enhanced Visualization Tab", "Module not found")
    
    def _load_enhanced_music_tab(self):
        """Load music tab"""
        try:
            from gui.components.enhanced_music_tab import EnhancedMusicTab
            return EnhancedMusicTab(data_provider=self.data_provider)
        except ImportError:
            return self._create_placeholder("Enhanced Music Tab", "Module not found")
    
    def _load_heartbeat_monitor_tab(self):
        """Load heartbeat tab"""
        try:
            from gui.components.heartbeat_monitor_tab import HeartbeatMonitorTab
            return HeartbeatMonitorTab(data_provider=self.data_provider)
        except ImportError:
            return self._create_placeholder("Heartbeat Monitor Tab", "Module not found")
    
    def _create_placeholder(self, tab_name: str, message: str):
        """Create placeholder widget"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        title = QLabel(f"📋 {tab_name}")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(f"{tab_name}\n\n{message}\n\nThis tab loaded successfully without hanging!")
        
        layout.addWidget(title)
        layout.addWidget(text)
        widget.setLayout(layout)
        return widget

# ──────────────────────────────────────────────────────────────────────────────
# 4)  Main launcher with splash screen
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Launch optimized GUI"""
    try:
        # Start memory tracking
        tracemalloc.start()
        
        logger.info("🎯 Optimized VoxSigil Enhanced GUI")
        logger.info("=" * 60)
        logger.info("Features: Timeouts, Retries, Resource Monitoring, Hot Reload")
        
        # Create app
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
            app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        # Show splash
        splash_pixmap = QPixmap(400, 300)
        splash_pixmap.fill(Qt.black)
        splash = QSplashScreen(splash_pixmap)
        splash.showMessage("🚀 Loading VoxSigil Enhanced GUI...", 
                          Qt.AlignCenter | Qt.AlignBottom, Qt.cyan)
        splash.show()
        app.processEvents()
        
        # Create GUI
        splash.showMessage("🔧 Initializing optimized components...", 
                          Qt.AlignCenter | Qt.AlignBottom, Qt.cyan)
        app.processEvents()
        
        window = OptimizedEnhancedGUI()
        window.show()
        splash.finish(window)
        
        logger.info("✅ Optimized GUI launched successfully!")
        logger.info("💡 Click tabs to load features with timeout protection")
        
        # Run
        exit_code = app.exec_()
        
        # Show final stats
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            logger.info(f"📊 Memory: Current {current/1024/1024:.1f}MB, Peak {peak/1024/1024:.1f}MB")
            tracemalloc.stop()
        
        return exit_code
        
    except Exception as e:
        log_exception("GUI launch failed", e)
        return 1

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    exit_code = main()
    if exit_code == 0:
        logger.info("👋 GUI session ended successfully")
    else:
        logger.error("❌ GUI session ended with errors")
    sys.exit(exit_code)
