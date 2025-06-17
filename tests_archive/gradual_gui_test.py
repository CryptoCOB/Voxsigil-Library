#!/usr/bin/env python3
"""
Enhanced GUI Gradual Test - Load components step by step
"""

import sys
import logging

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("GradualGUI")

def main():
    logger.info("Starting Enhanced GUI Gradual Test...")
    
    try:
        # 1. Import PyQt5
        from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QPushButton
        from PyQt5.QtCore import QTimer
        
        # 2. Create application
        app = QApplication(sys.argv)
        
        # 3. Create main window
        window = QMainWindow()
        window.setWindowTitle("VoxSigil Enhanced GUI - Step by Step Test")
        window.resize(1000, 700)
        
        # 4. Create central widget
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Status label
        status_label = QLabel("Starting GUI components...")
        status_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px; background: lightblue;")
        main_layout.addWidget(status_label)
        
        # Tab widget
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # Test button
        test_button = QPushButton("Test Real-Time Data Provider")
        main_layout.addWidget(test_button)
        
        central_widget.setLayout(main_layout)
        window.setCentralWidget(central_widget)
        
        # 5. Add basic tabs first
        basic_tabs = []
        for tab_name in ["Model", "Training", "Music", "Visualization", "Dashboard"]:
            tab = QWidget()
            layout = QVBoxLayout()
            
            label = QLabel(f"{tab_name} Tab\n\nBasic tab created. Enhanced components will be loaded step by step.")
            label.setStyleSheet("font-size: 12px; padding: 20px;")
            layout.addWidget(label)
            
            tab.setLayout(layout)
            tab_widget.addTab(tab, tab_name)
            basic_tabs.append((tab, layout, label))
        
        # 6. Show window
        window.show()
        status_label.setText("✅ Basic GUI created and shown!")
        
        # 7. Define step-by-step loading functions
        step = [0]  # Use list to allow modification in nested functions
        
        def next_step():
            try:
                step[0] += 1
                current_step = step[0]
                
                if current_step == 1:
                    status_label.setText("Step 1: Testing Real-Time Data Provider...")
                    from gui.components.real_time_data_provider import RealTimeDataProvider
                    provider = RealTimeDataProvider()
                    status_label.setText("✅ Step 1: Real-Time Data Provider imported!")
                    
                    # Schedule next step
                    QTimer.singleShot(1000, next_step)
                    
                elif current_step == 2:
                    status_label.setText("Step 2: Getting system metrics...")
                    from gui.components.real_time_data_provider import RealTimeDataProvider
                    provider = RealTimeDataProvider()
                    metrics = provider.get_system_metrics()
                    
                    # Update first tab
                    first_tab, first_layout, first_label = basic_tabs[0]
                    system_info = QLabel(f"Real System Data:\nCPU: {metrics.get('cpu_percent', 0):.1f}%\nMemory: {metrics.get('memory_percent', 0):.1f}%")
                    system_info.setStyleSheet("color: green; font-weight: bold; margin: 10px;")
                    first_layout.addWidget(system_info)
                    
                    status_label.setText("✅ Step 2: System metrics loaded!")
                    QTimer.singleShot(1000, next_step)
                    
                elif current_step == 3:
                    status_label.setText("Step 3: Testing enhanced tab imports...")
                    try:
                        from gui.components.enhanced_model_tab import EnhancedModelTab
                        status_label.setText("✅ Step 3: Enhanced Model Tab imported!")
                        QTimer.singleShot(1000, next_step)
                    except Exception as e:
                        status_label.setText(f"❌ Step 3 failed: {e}")
                        
                elif current_step == 4:
                    status_label.setText("Step 4: All tests completed! GUI is working.")
                    
            except Exception as e:
                status_label.setText(f"❌ Step {current_step} failed: {e}")
                logger.error(f"Step {current_step} error: {e}")
        
        # 8. Connect test button
        test_button.clicked.connect(next_step)
        
        # 9. Start first step after 2 seconds
        QTimer.singleShot(2000, next_step)
        
        # 10. Run the application
        return app.exec_()
        
    except Exception as e:
        logger.error(f"Gradual GUI failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
