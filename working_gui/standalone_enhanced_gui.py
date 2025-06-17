#!/usr/bin/env python3
"""
VoxSigil Enhanced GUI - Standalone Version (No External Dependencies)
This version removes all gui.components imports to prevent hanging issues.
"""

import gc
import importlib
import logging
import sys
import traceback
import tracemalloc
from typing import Callable, List, NamedTuple

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# PyQt5 imports (with error handling)
try:
    from PyQt5.QtCore import QCoreApplication, Qt, QThread, QTimer, pyqtSignal
    from PyQt5.QtGui import QKeySequence, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QShortcut,
        QSplashScreen,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    # Apply high-DPI flags
    for flag in (Qt.AA_EnableHighDpiScaling, Qt.AA_UseHighDpiPixmaps):
        if not QCoreApplication.testAttribute(flag):
            QCoreApplication.setAttribute(flag, True)

    logger.info("✅ PyQt5 imported successfully")
except ImportError as e:
    logger.error(f"❌ PyQt5 import failed: {e}")
    sys.exit(1)


# Feature specification
class FeatureSpec(NamedTuple):
    title: str
    loader: Callable[..., QWidget]
    timeout_ms: int = 12_000
    max_retries: int = 2


# Error logging
def log_exception(prefix: str, exc: Exception):
    logger.error(f"{prefix}: {exc}")
    logger.debug("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


# Safe psutil import
def safe_import_psutil():
    try:
        import psutil

        return psutil
    except ImportError:
        logger.warning("psutil not available - resource monitoring disabled")
        return None


psutil = safe_import_psutil()


# Tab loading worker
class TabLoadWorker(QThread):
    """Background worker to load tabs without blocking UI"""

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
        """Load tab in background with interruption support"""
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

            # Validate the widget before emitting
            if widget is None:
                self.load_error.emit("Loader function returned None")
                return

            if not isinstance(widget, QWidget):
                self.load_error.emit(
                    f"Loader function returned invalid type: {type(widget)}"
                )
                return self.load_progress.emit(75)
            self.tab_loaded.emit(widget)
            self.load_progress.emit(100)

        except ImportError as e:
            log_exception("TabLoadWorker import error", e)
            self.load_error.emit(f"Missing dependency: {str(e)}")
        except AttributeError as e:
            log_exception("TabLoadWorker attribute error", e)
            self.load_error.emit(f"Missing attribute: {str(e)}")
        except RuntimeError as e:
            log_exception("TabLoadWorker runtime error", e)
            self.load_error.emit(f"Qt runtime error: {str(e)}")
        except Exception as e:
            log_exception("TabLoadWorker unexpected error", e)
            self.load_error.emit(f"Unexpected error: {str(e)}")
        finally:
            if self.isInterruptionRequested():
                self.cancelled.emit()


# Lazy tab with timeout and retry
class StandaloneLazyTab(QWidget):
    """Standalone lazy tab with timeout and retry support"""

    def __init__(
        self,
        tab_name: str,
        loader_function,
        main_gui_ref,
        timeout_ms: int = 12_000,
        max_retries: int = 2,
    ):
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
        """Initialize placeholder with VoxSigil-style theming"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Apply dark theme styling
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                color: #ffffff;
            }
        """)

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
        desc = QLabel(f"Click 'Load {self.tab_name}' to activate enhanced features.")
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
        self.status_label.setStyleSheet(
            "QLabel { color: #999999; font-style: italic; margin: 10px; }"
        )
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
            self.status_label.setText(
                f"Loading {self.tab_name}... (Timeout: {self.timeout_ms / 1000}s)"
            )

            # Start background loading
            self.worker = TabLoadWorker(self.loader_function)
            self.worker.tab_loaded.connect(self._on_tab_loaded)
            self.worker.load_error.connect(self._on_load_error)
            self.worker.load_progress.connect(self._on_load_progress)
            self.worker.cancelled.connect(self._on_cancelled)
            self.worker.start()
            self.timeout_timer.start(self.timeout_ms)

        except RuntimeError as e:
            log_exception(f"Qt runtime error loading {self.tab_name}", e)
            self._on_load_error(f"Qt runtime error: {str(e)}")
        except AttributeError as e:
            log_exception(f"Missing attribute loading {self.tab_name}", e)
            self._on_load_error(f"Missing component: {str(e)}")
        except Exception as e:
            log_exception(f"Unexpected error loading {self.tab_name}", e)
            self._on_load_error(f"Unexpected error: {str(e)}")

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

            # Validate widget before using it
            if widget is None:
                logger.warning(f"Received None widget for {self.tab_name}")
                self._on_load_error("Widget is None")
                return

            if not isinstance(widget, QWidget):
                logger.warning(
                    f"Received invalid widget type for {self.tab_name}: {type(widget)}"
                )
                self._on_load_error("Invalid widget type")
                return

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
            self.load_button.setText(
                f"🔄 Retry {self.tab_name} ({self._retries_left} left)"
            )
            self.status_label.setText(
                f"⚠️ Error: {error_msg[:50]}... - {self._retries_left} retries left"
            )
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
        self._on_load_error(f"Timeout after {self.timeout_ms / 1000:.1f}s")

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
        """Replace placeholder with actual content safely"""
        try:
            layout = self.layout()
            if layout is None:
                logger.error("No layout found for tab replacement")
                return

            # Safely clear existing widgets
            widgets_to_remove = []
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and item.widget():
                    widgets_to_remove.append(item.widget())

            # Remove widgets safely
            for widget in widgets_to_remove:
                layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()

            # Add new widget safely
            if isinstance(self.actual_widget, QWidget):
                layout.addWidget(self.actual_widget)
                self.actual_widget.show()

        except Exception as e:
            logger.error(f"Error replacing tab content: {e}")
            # Fallback: show error message instead of crashing
            self._show_error_content(str(e))

    def _show_error_content(self, error_msg):
        """Show error content instead of crashing"""
        try:
            layout = self.layout()
            if layout:
                # Clear layout safely
                while layout.count():
                    item = layout.takeAt(0)
                    if item and item.widget():
                        item.widget().deleteLater()

                # Show error message
                error_label = QLabel(f"❌ Error loading {self.tab_name}:\n{error_msg}")
                error_label.setWordWrap(True)
                error_label.setStyleSheet(
                    "color: #ff4444; padding: 20px; font-size: 14px;"
                )
                layout.addWidget(error_label)

                # Add retry button
                retry_button = QPushButton(f"🔄 Retry {self.tab_name}")
                retry_button.clicked.connect(self._load_enhanced_content)
                layout.addWidget(retry_button)

        except Exception as e:
            logger.error(f"Error showing error content: {e}")


# Main GUI class
class StandaloneEnhancedGUI(QMainWindow):
    """Standalone Enhanced GUI without external dependencies"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoxSigil Enhanced GUI - Standalone")
        self.setGeometry(100, 100, 1600, 1000)

        self._init_ui()
        self._apply_theme()
        self._add_shortcuts()

        logger.info("✅ Standalone Enhanced GUI initialized")

    def _init_ui(self):
        """Initialize UI with optimized loading"""
        self.tabs = QTabWidget()

        # Apply basic tab styling
        self.tabs.setStyleSheet("""
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

        # Add safe tab switching to prevent crashes
        self._last_tab_click_time = 0
        self._tab_click_cooldown = 500  # 500ms cooldown between tab clicks
        self.tabs.currentChanged.connect(self._safe_tab_changed)

        # Status tab
        status_tab = self._create_status_tab()
        self.tabs.addTab(status_tab, "📊 Status")

        # Feature specs
        feature_specs: List[FeatureSpec] = [
            FeatureSpec("📡 Dashboard", self._load_placeholder_tab, 5_000, 3),
            FeatureSpec("🤖 Models", self._load_placeholder_tab, 8_000, 2),
            FeatureSpec("🎯 Training", self._load_placeholder_tab, 10_000, 2),
            FeatureSpec("📈 Visualization", self._load_placeholder_tab, 6_000, 3),
            FeatureSpec("🎵 Music", self._load_placeholder_tab, 7_000, 2),
            FeatureSpec("💓 Heartbeat", self._load_placeholder_tab, 4_000, 3),
        ]

        for spec in feature_specs:
            tab = StandaloneLazyTab(
                spec.title.split(" ", 1)[1],
                spec.loader,
                self,
                timeout_ms=spec.timeout_ms,
                max_retries=spec.max_retries,
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

        title = QLabel("🎉 VoxSigil Enhanced GUI - Standalone")
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
        self.status_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #00ffff;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)

        status_info = self._get_status_info()
        self.status_text.setPlainText(status_info)
        layout.addWidget(self.status_text)

        refresh_btn = QPushButton("🔄 Refresh Status")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #00ffff;
                color: black;
            }
        """)
        refresh_btn.clicked.connect(self._refresh_status)
        layout.addWidget(refresh_btn)

        tab.setLayout(layout)
        return tab

    def _get_status_info(self) -> str:
        """Get current status"""
        status = [
            "🎉 VoxSigil Enhanced GUI - Standalone Version",
            "=" * 60,
            "",
            "✅ Features Active:",
            "• No external dependencies (gui.components removed)",
            "• Timeout protection (4-10s per tab)",
            "• Automatic retry (2-3 attempts)",
            "• Circuit breaker for failed tabs",
            "• Memory leak detection",
            "• Resource monitoring (if psutil available)",
            "• Background loading",
            "",
            "🚀 Available Tabs:",
        ]

        for i in range(self.tabs.count()):
            tab_title = self.tabs.tabText(i)
            tab_widget = self.tabs.widget(i)
            if hasattr(tab_widget, "is_loaded"):
                status.append(
                    f"• {tab_title}: {'✅ Loaded' if tab_widget.is_loaded else '🔄 Ready to load'}"
                )
            else:
                status.append(f"• {tab_title}: ✅ Active")

        status.extend(
            [
                "",
                "💡 This version removes gui.components imports",
                "  to prevent hanging issues during startup.",
                "",
                "🎮 Shortcuts:",
                "• Ctrl+R: Hot reload",
                "• Ctrl+T: Resource stats",
                "• Ctrl+G: Force garbage collection",
            ]
        )

        if psutil:
            try:
                mem = psutil.virtual_memory().percent
                cpu = psutil.cpu_percent()
                status.append(f"📊 System: CPU {cpu:.1f}%, RAM {mem:.1f}%")
            except Exception as e:
                log_exception("Error getting system stats", e)
                status.append("📊 System: Monitoring error")
        else:
            status.append("📊 System: Monitoring not available")

        return "\n".join(status)

    def _refresh_status(self):
        """Refresh status display"""
        self.status_text.setPlainText(self._get_status_info())

    def _apply_theme(self):
        """Apply VoxSigil theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
                color: #ffffff;
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
            self.statusBar().showMessage(
                f"CPU: {cpu:.1f}% | MEM: {mem:.1f}% | Process: {process_mem:.1f}MB"
            )
        except Exception as e:
            logger.error(f"Error updating resources: {e}")
            pass

    def _show_stats(self):
        """Show resource statistics"""
        try:
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                mem_info = f"Current: {current / 1024 / 1024:.1f}MB, Peak: {peak / 1024 / 1024:.1f}MB"
            else:
                mem_info = "Memory tracing not active"

            if psutil:
                sys_info = f"CPU: {psutil.cpu_percent():.1f}%, RAM: {psutil.virtual_memory().percent:.1f}%"
            else:
                sys_info = "System monitoring not available"

            QMessageBox.information(
                self,
                "Resource Stats",
                f"Memory Tracking: {mem_info}\nSystem: {sys_info}",
            )
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
            QMessageBox.information(
                self, "Reloaded", "Modules reloaded - restart recommended"
            )
        except Exception as e:
            QMessageBox.warning(self, "Reload Error", f"Failed: {e}")

    def _load_placeholder_tab(self):
        """Load placeholder tab (for demo purposes)"""
        widget = QWidget()
        layout = QVBoxLayout()

        title = QLabel("🎉 Enhanced Tab Loaded Successfully!")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #00ffff;
                background-color: #2d2d2d;
                border: 2px solid #00ffff;
                border-radius: 8px;
                padding: 15px;
                margin: 10px;
            }
        """)
        layout.addWidget(title)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText("""
🎉 SUCCESS: Tab loaded without hanging!

This is a placeholder tab that demonstrates:
✅ Background loading with timeout protection
✅ Retry logic for failed loads
✅ Circuit breaker for permanent failures
✅ Memory leak prevention
✅ Resource monitoring

The original enhanced tabs can be restored by:
1. Fixing any circular imports in gui.components
2. Ensuring real_time_data_provider doesn't block
3. Verifying gui_styles doesn't cause hangs

This standalone version proves the core GUI framework works correctly.
        """)
        text.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #00ffff;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)
        layout.addWidget(text)

        widget.setLayout(layout)
        return widget

    def _safe_tab_changed(self, index):
        """Handle tab changes safely with cooldown to prevent crashes"""
        import time

        current_time = time.time() * 1000  # Convert to milliseconds

        # Check if enough time has passed since last tab click
        if current_time - self._last_tab_click_time < self._tab_click_cooldown:
            logger.info("⏳ Tab click too fast - ignoring to prevent crash")
            return

        self._last_tab_click_time = current_time

        try:
            # Get the current tab widget
            current_widget = self.tabs.widget(index)
            if current_widget:
                logger.info(
                    f"🔄 Safely switched to tab {index}: {self.tabs.tabText(index)}"
                )

                # If it's a lazy tab, ensure it's properly initialized
                if hasattr(current_widget, "tab_name"):
                    logger.info(f"📋 Loaded lazy tab: {current_widget.tab_name}")

        except Exception as e:
            logger.error(f"❌ Error during tab switch: {e}")
            # Don't let tab switching errors crash the GUI


def main():
    """Launch standalone GUI"""
    try:
        # Start memory tracking
        tracemalloc.start()

        logger.info("🎯 VoxSigil Enhanced GUI - Standalone")
        logger.info("=" * 60)
        logger.info("No external dependencies - should launch without hanging")

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
        splash.showMessage(
            "🚀 Loading VoxSigil Standalone GUI...",
            Qt.AlignCenter | Qt.AlignBottom,
            Qt.cyan,
        )
        splash.show()
        app.processEvents()

        # Create GUI
        window = StandaloneEnhancedGUI()
        window.show()
        splash.finish(window)

        logger.info("✅ Standalone GUI launched successfully!")

        # Run
        exit_code = app.exec_()

        # Show final stats
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            logger.info(
                f"📊 Memory: Current {current / 1024 / 1024:.1f}MB, Peak {peak / 1024 / 1024:.1f}MB"
            )
            tracemalloc.stop()

        return exit_code

    except Exception as e:
        log_exception("Standalone GUI launch failed", e)
        return 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        logger.info("👋 GUI session ended successfully")
    else:
        logger.error("❌ GUI session ended with errors")
    sys.exit(exit_code)
