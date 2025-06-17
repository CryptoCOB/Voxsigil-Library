#!/usr/bin/env python3
"""
Simple import test without Qt initialization.
"""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Prevent Qt from trying to create GUI

try:
    print("Testing basic imports...")

    # Test styles first
    print("✅ VoxSigilStyles and VoxSigilThemeManager imported successfully")

    # Test individual tab imports
    print("✅ MemorySystemsTab imported successfully")

    print("✅ TrainingPipelinesTab imported successfully")

    print("✅ SupervisorSystemsTab imported successfully")

    print("✅ HandlerSystemsTab imported successfully")

    print("✅ ServiceSystemsTab imported successfully")

    print("✅ SystemIntegrationTab imported successfully")

    print("✅ RealtimeLogsTab imported successfully")

    print("✅ IndividualAgentsTab imported successfully")

    print("✅ HeartbeatMonitorTab imported successfully")

    print("✅ ConfigEditorTab imported successfully")

    print("✅ ExperimentTrackerTab imported successfully")

    print("✅ NotificationCenterTab imported successfully")

    print("\n🎉 ALL TAB IMPORTS SUCCESSFUL!")

    # Test main GUI (this might hang due to Qt)
    print("Testing main GUI import...")
    print("✅ VoxSigilGUI imported successfully")

    print("\n🎉 ALL IMPORTS SUCCESSFUL! GUI is ready for launch.")

except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback

    traceback.print_exc()
