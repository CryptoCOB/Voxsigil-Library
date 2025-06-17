#!/usr/bin/env python3
"""
VoxSigil Enhanced GUI - Crash-Proof Version
This version uses safe placeholder tabs and won't crash when clicking tabs.
"""

import logging
import sys
import gc
import tracemalloc
import traceback
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# PyQt5 imports with error handling
try:
    from PyQt5.QtWidgets import (
        QApplication, QLabel, QMainWindow, QPushButton, QTabWidget,
        QVBoxLayout, QWidget, QTextEdit, QProgressBar, QHBoxLayout,
        QSplashScreen, QMessageBox, QShortcut, QScrollArea
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
    from PyQt5.QtGui import QPixmap, QKeySequence, QFont
    
    logger.info("✅ PyQt5 imported successfully")
except ImportError as e:
    logger.error(f"❌ PyQt5 import failed: {e}")
    sys.exit(1)

# Safe resource monitoring
def safe_import_psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        logger.warning("psutil not available - resource monitoring disabled")
        return None

psutil = safe_import_psutil()

class SafeTabLoader(QThread):
    """Safe tab loader that won't crash the application"""
    
    tab_loaded = pyqtSignal(QWidget)
    load_error = pyqtSignal(str)
    load_progress = pyqtSignal(int)
    
    def __init__(self, tab_name: str, loader_function):
        super().__init__()
        self.tab_name = tab_name
        self.loader_function = loader_function
        
    def run(self):
        """Load tab safely with error handling"""
        try:
            self.load_progress.emit(25)
            
            # Attempt to load the tab
            widget = self.loader_function()
            
            self.load_progress.emit(75)
            self.tab_loaded.emit(widget)
            self.load_progress.emit(100)
            
        except Exception as e:
            error_msg = f"Failed to load {self.tab_name}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            self.load_error.emit(error_msg)

class CrashProofLazyTab(QWidget):
    """A tab that won't crash when clicked"""
    
    def __init__(self, tab_name: str, tab_description: str, main_gui_ref):
        super().__init__()
        self.tab_name = tab_name
        self.tab_description = tab_description
        self.main_gui_ref = main_gui_ref
        self.is_loaded = False
        self.actual_widget = None
        self.worker = None
        
        self._init_safe_placeholder()
    
    def _init_safe_placeholder(self):
        """Initialize safe placeholder that won't crash"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Apply VoxSigil styling
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                color: #ffffff;
            }
        """)
        
        # Title
        title = QLabel(f"🎯 {self.tab_name}")
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
        desc = QLabel(self.tab_description)
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("QLabel { font-size: 14px; color: #cccccc; padding: 10px; }")
        layout.addWidget(desc)
        
        # Load button
        self.load_button = QPushButton(f"🚀 Load {self.tab_name}")
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
        self.load_button.clicked.connect(self._load_safe_content)
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
    
    def _load_safe_content(self):
        """Load content safely without crashing"""
        if self.is_loaded or self.worker is not None:
            return
            
        try:
            # Show loading state
            self.load_button.setEnabled(False)
            self.load_button.setText("🔄 Loading...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.status_label.setText(f"Loading {self.tab_name}...")
            
            # Create safe demo content instead of trying to import problematic modules
            self.worker = SafeTabLoader(self.tab_name, self._create_demo_content)
            self.worker.tab_loaded.connect(self._on_tab_loaded)
            self.worker.load_error.connect(self._on_load_error)
            self.worker.load_progress.connect(self._on_load_progress)
            
            self.worker.start()
            
        except Exception as e:
            logger.error(f"Error starting tab load: {e}")
            self._on_load_error(str(e))
    
    def _create_demo_content(self):
        """Create safe demo content that won't crash"""
        # Create a demo widget with actual functionality
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(f"🎉 {self.tab_name} - Demo Version")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
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
        
        # Create scrollable content area
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        # Add demo content based on tab type
        demo_content = self._get_demo_content_for_tab()
        
        content_text = QTextEdit()
        content_text.setReadOnly(True)
        content_text.setPlainText(demo_content)
        content_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #00ffff;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }
        """)
        scroll_layout.addWidget(content_text)
        
        # Add some interactive elements
        if self.tab_name in ["Dashboard", "Models", "Training"]:
            self._add_interactive_demo_elements(scroll_layout)
        
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Status info
        status_info = QLabel(f"✅ {self.tab_name} loaded successfully (Demo version)")
        status_info.setAlignment(Qt.AlignCenter)
        status_info.setStyleSheet("QLabel { color: #00ff00; font-weight: bold; margin: 10px; }")
        layout.addWidget(status_info)
        
        widget.setLayout(layout)
        return widget
    
    def _get_demo_content_for_tab(self):
        """Get demo content specific to each tab"""
        if self.tab_name == "Dashboard":
            return """
🔄 LIVE DASHBOARD - Demo Mode

📊 System Metrics:
- CPU Usage: 45.2%
- Memory Usage: 62.8%
- GPU Usage: 23.1%
- Network I/O: 1.2 MB/s

🚀 Active Processes:
- VoxSigil Engine: Running
- Neural Networks: 3 models loaded
- Training Pipeline: Idle
- Data Processing: Active

📈 Performance Stats:
- Inference Speed: 150ms avg
- Training Accuracy: 94.7%
- Model Efficiency: 87%
- System Uptime: 2h 34m

🔧 Recent Activities:
- Model training completed
- New dataset processed
- Performance optimization applied
- System health check passed

This is a demo dashboard showing how real-time data would be displayed.
The actual dashboard would connect to live VoxSigil metrics.
            """
        elif self.tab_name == "Models":
            return """
🤖 MODEL MANAGEMENT - Demo Mode

📋 Available Models:
1. VoxSigil-Base-v2.1
   - Type: Neural TTS
   - Status: Ready
   - Accuracy: 94.2%
   - Size: 1.2GB

2. VoxSigil-Enhanced-v1.8
   - Type: Voice Synthesis
   - Status: Training
   - Progress: 67%
   - ETA: 2h 15m

3. VoxSigil-Lite-v3.0
   - Type: Lightweight TTS
   - Status: Ready
   - Accuracy: 91.8%
   - Size: 320MB

🔧 Model Operations:
- Load/Unload models
- Train new models
- Fine-tune existing models
- Export model weights
- Performance benchmarking

📊 Training Metrics:
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 150/200
- Loss: 0.0234
- Validation Accuracy: 92.1%

This demo shows model management interface.
Real version would control actual VoxSigil models.
            """
        elif self.tab_name == "Training":
            return """
🎯 TRAINING PIPELINE - Demo Mode

🚀 Active Training Jobs:
1. Voice Style Transfer
   - Progress: 78%
   - ETA: 1h 42m
   - Loss: 0.0156
   - Accuracy: 89.3%

2. Accent Adaptation
   - Progress: 23%
   - ETA: 4h 18m
   - Loss: 0.0892
   - Accuracy: 76.8%

📊 Training Configuration:
- Optimizer: AdamW
- Learning Rate: 0.0001
- Batch Size: 16
- Data Augmentation: Enabled
- Mixed Precision: Enabled

📈 Performance Metrics:
- GPU Utilization: 94%
- Memory Usage: 7.2/11GB
- Training Speed: 23 samples/sec
- Validation Frequency: Every 100 steps

🔧 Pipeline Controls:
- Start/Stop training
- Adjust hyperparameters
- Save checkpoints
- Resume from checkpoint
- Export trained models

This demo shows training pipeline management.
Real version would control actual VoxSigil training.
            """
        elif self.tab_name == "Visualization":
            return """
📈 VISUALIZATION - Demo Mode

🎨 Available Visualizations:
- Training Loss Curves
- Accuracy Metrics
- Model Architecture Diagrams
- Data Flow Visualizations
- Performance Heatmaps

📊 Real-time Charts:
- Loss vs Epochs
- Learning Rate Schedule
- Gradient Flow
- Activation Distributions
- Weight Histograms

🔍 Analysis Tools:
- Model Interpretability
- Feature Importance
- Error Analysis
- Performance Profiling
- Resource Usage Graphs

This demo shows visualization capabilities.
Real version would display live VoxSigil metrics and charts.
            """
        elif self.tab_name == "Music":
            return """
🎵 MUSIC GENERATION - Demo Mode

🎼 Music Models:
- VoxSigil-Music-v1.2
- Melody Generator
- Harmony Assistant
- Rhythm Synthesizer

🎹 Generation Controls:
- Genre Selection
- Tempo Control
- Key Signature
- Instrument Selection
- Style Parameters

📝 Recent Compositions:
1. Classical Piano Piece
2. Jazz Improvisation
3. Electronic Ambient
4. Folk Guitar Melody

This demo shows music generation interface.
Real version would generate actual music using VoxSigil.
            """
        elif self.tab_name == "Heartbeat":
            return """
💓 HEARTBEAT MONITOR - Demo Mode

🔄 System Health:
- Core Engine: ✅ Healthy
- Neural Networks: ✅ Operational  
- Data Pipeline: ✅ Active
- API Services: ✅ Responsive

📊 Performance Metrics:
- Response Time: 45ms
- Throughput: 1,250 req/min
- Error Rate: 0.02%
- Uptime: 99.98%

🚨 Alerts & Notifications:
- No critical issues
- 2 minor warnings
- Last check: 30 seconds ago

This demo shows system monitoring.
Real version would track actual VoxSigil health.
            """
        else:
            return f"""
📋 {self.tab_name.upper()} - Demo Mode

This is a demonstration of the {self.tab_name} tab.

🎯 Features:
- Enhanced functionality
- Real-time updates
- Interactive controls
- Performance monitoring

✅ Status: Demo mode active
🔧 Note: This shows how the real {self.tab_name} tab would work.

In the full version, this tab would provide:
- Live data from VoxSigil systems
- Interactive controls and settings
- Real-time performance metrics
- Advanced configuration options

The demo version is safe and won't crash your system.
            """
    
    def _add_interactive_demo_elements(self, layout):
        """Add interactive elements to demo tabs"""
        # Add some buttons for interactivity
        button_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 Refresh Data")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        refresh_btn.clicked.connect(self._demo_refresh)
        button_layout.addWidget(refresh_btn)
        
        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        settings_btn.clicked.connect(self._demo_settings)
        button_layout.addWidget(settings_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def _demo_refresh(self):
        """Demo refresh action"""
        QMessageBox.information(self, "Demo Action", f"🔄 {self.tab_name} data refreshed!\n\nIn the real version, this would update live data from VoxSigil systems.")
    
    def _demo_settings(self):
        """Demo settings action"""
        QMessageBox.information(self, "Demo Action", f"⚙️ {self.tab_name} settings opened!\n\nIn the real version, this would show configuration options for {self.tab_name}.")
    
    def _on_load_progress(self, progress):
        """Update progress"""
        self.progress_bar.setValue(progress)
        if progress == 25:
            self.status_label.setText("Creating demo content...")
        elif progress == 75:
            self.status_label.setText("Finalizing interface...")
        elif progress == 100:
            self.status_label.setText("Loading complete!")
    
    def _on_tab_loaded(self, widget):
        """Handle successful load"""
        try:
            self.actual_widget = widget
            self._replace_with_content()
            self.is_loaded = True
            self.worker = None
            
            # Micro GC
            QTimer.singleShot(0, gc.collect)
            
            logger.info(f"✅ {self.tab_name} demo loaded successfully")
            
        except Exception as e:
            logger.error(f"Error displaying {self.tab_name}: {e}")
            self._on_load_error(f"Display error: {e}")
    
    def _on_load_error(self, error_msg):
        """Handle load errors"""
        logger.error(f"Failed to load {self.tab_name}: {error_msg}")
        
        self.progress_bar.setVisible(False)
        self.load_button.setEnabled(True)
        self.load_button.setText(f"❌ {self.tab_name} Failed")
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: #ff4444;")
        self.worker = None
    
    def _replace_with_content(self):
        """Replace placeholder with actual content"""
        layout = self.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if isinstance(self.actual_widget, QWidget):
            layout.addWidget(self.actual_widget)

class CrashProofEnhancedGUI(QMainWindow):
    """Crash-proof enhanced GUI that won't crash when clicking tabs"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoxSigil Enhanced GUI - Crash-Proof")
        self.setGeometry(100, 100, 1600, 1000)
        
        self._init_ui()
        self._apply_theme()
        self._add_shortcuts()
        
        logger.info("✅ Crash-proof Enhanced GUI initialized")
    
    def _init_ui(self):
        """Initialize UI"""
        self.tabs = QTabWidget()
        
        # Apply tab styling
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
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #00ffff;
                color: #000000;
            }
            QTabBar::tab:hover {
                background-color: #555555;
            }
        """)
        
        # Status tab
        status_tab = self._create_status_tab()
        self.tabs.addTab(status_tab, "📊 Status")
        
        # Safe demo tabs that won't crash
        safe_tabs = [
            ("📡 Dashboard", "Real-time system monitoring and metrics dashboard"),
            ("🤖 Models", "Neural network model management and training interface"),
            ("🎯 Training", "Machine learning training pipeline and progress tracking"),
            ("📈 Visualization", "Data visualization and performance analytics"),
            ("🎵 Music", "AI music generation and composition tools"),
            ("💓 Heartbeat", "System health monitoring and alerting"),
        ]
        
        for tab_title, tab_description in safe_tabs:
            tab_name = tab_title.split(" ", 1)[1]
            tab = CrashProofLazyTab(tab_name, tab_description, self)
            self.tabs.addTab(tab, tab_title)
        
        self.setCentralWidget(self.tabs)
        
        # Resource monitoring
        if psutil:
            self._resource_timer = QTimer(self)
            self._resource_timer.timeout.connect(self._update_resources)
            self._resource_timer.start(3_000)
        
        logger.info(f"✅ GUI with {self.tabs.count()} crash-proof tabs ready")
    
    def _create_status_tab(self):
        """Create status tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        title = QLabel("🛡️ VoxSigil Enhanced GUI - Crash-Proof Version")
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
                background-color: #45a049;
            }
        """)
        refresh_btn.clicked.connect(self._refresh_status)
        layout.addWidget(refresh_btn)
        
        tab.setLayout(layout)
        return tab
    
    def _get_status_info(self) -> str:
        """Get current status"""
        status = [
            "🛡️ VoxSigil Enhanced GUI - Crash-Proof Version",
            "=" * 60,
            "",
            "✅ Safety Features Active:",
            "• Safe tab loading (no crashes)",
            "• Demo content instead of problematic imports",
            "• Error handling and recovery",
            "• Memory leak prevention",
            "• Resource monitoring",
            "",
            "🎯 Available Tabs:",
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
            "💡 How it works:",
            "• Click any tab to see demo content",
            "• Tabs load safely without crashing",
            "• Demo shows what real functionality would look like",
            "• All features are interactive but safe",
            "",
            "🔧 This version prevents crashes by:",
            "• Not importing problematic gui.components modules",
            "• Using safe demo content instead",
            "• Comprehensive error handling",
            "• Memory management and cleanup",
            "",
            "🎮 Keyboard Shortcuts:",
            "• Ctrl+R: Refresh GUI",
            "• Ctrl+G: Force garbage collection",
        ])
        
        if psutil:
            try:
                mem = psutil.virtual_memory().percent
                cpu = psutil.cpu_percent()
                status.extend([
                    "",
                    f"📊 System Resources: CPU {cpu:.1f}%, RAM {mem:.1f}%",
                ])
            except:
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
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._refresh_gui)
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
    
    def _refresh_gui(self):
        """Refresh GUI"""
        self._refresh_status()
        QMessageBox.information(self, "Refreshed", "🔄 GUI refreshed successfully!")
    
    def _force_gc(self):
        """Force garbage collection"""
        collected = gc.collect()
        self.statusBar().showMessage(f"GC: Collected {collected} objects", 3000)
        logger.info(f"🗑️ GC collected {collected} objects")

def main():
    """Launch crash-proof GUI"""
    try:
        # Start memory tracking
        tracemalloc.start()
        
        logger.info("🛡️ VoxSigil Crash-Proof Enhanced GUI")
        logger.info("=" * 60)
        logger.info("Safe demo version - tabs won't crash when clicked")
        
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
        splash.showMessage("🛡️ Loading Crash-Proof VoxSigil GUI...", 
                          Qt.AlignCenter | Qt.AlignBottom, Qt.cyan)
        splash.show()
        app.processEvents()
        
        # Create GUI
        window = CrashProofEnhancedGUI()
        window.show()
        splash.finish(window)
        
        logger.info("✅ Crash-proof GUI launched successfully!")
        logger.info("💡 Click any tab - they won't crash!")
        
        # Run
        exit_code = app.exec_()
        
        # Show final stats
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            logger.info(f"📊 Memory: Current {current/1024/1024:.1f}MB, Peak {peak/1024/1024:.1f}MB")
            tracemalloc.stop()
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Crash-proof GUI launch failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        logger.info("👋 GUI session ended successfully")
    else:
        logger.error("❌ GUI session ended with errors")
    sys.exit(exit_code)
