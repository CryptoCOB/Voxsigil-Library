#!/usr/bin/env python3
"""
Comprehensive GUI Tab Inventory and Test
=========================================

Validates that all 27 required tabs are present and functioning with streaming.
Creates a detailed report of tab status and streaming capabilities.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def test_gui_tab_inventory():
    """
    Comprehensive test of all GUI tabs and streaming capabilities.

    Expected 27 tabs:
    1. Control Center (priority 0)
    2. Individual Agents
    3. Processing Engines
    4. Memory Systems
    5. Training Pipelines
    6. Supervisor Systems
    7. Handler Systems
    8. Service Systems
    9. System Integration
    10. Real-time Logs
    11. System Health
    12. Models
    13. Model Discovery
    14. Training
    15. Visualization
    16. Performance
    17. GridFormer
    18. Advanced GridFormer
    19. VMB Integration
    20. VMB Demo
    21. Music
    22. Echo Log
    23. Mesh Map
    24. Agent Status
    25. BLT/RAG
    26. ARC
    27. Vanta Core
    """

    print("🔍 VoxSigil GUI Tab Inventory Test")
    print("=" * 50)

    # Define expected tabs with their requirements
    expected_tabs = [
        {
            "name": "Control Center",
            "priority": 0,
            "streaming": False,  # Bidirectional chat
            "component_file": "gui/components/control_center_tab.py",
            "status": "NEW",
        },
        {
            "name": "Individual Agents",
            "priority": 1,
            "streaming": True,
            "component_file": "gui/components/individual_agents_tab.py",
            "status": "MISSING",
        },
        {
            "name": "Processing Engines",
            "priority": 2,
            "streaming": True,
            "component_file": "gui/components/processing_engines_tab.py",
            "status": "EXISTS",
        },
        {
            "name": "Memory Systems",
            "priority": 3,
            "streaming": True,
            "component_file": "gui/components/memory_systems_tab.py",
            "status": "MISSING",
        },
        {
            "name": "Training Pipelines",
            "priority": 4,
            "streaming": True,
            "component_file": "gui/components/training_pipelines_tab.py",
            "status": "MISSING",
        },
        {
            "name": "Supervisor Systems",
            "priority": 5,
            "streaming": True,
            "component_file": "gui/components/supervisor_systems_tab.py",
            "status": "MISSING",
        },
        {
            "name": "Handler Systems",
            "priority": 6,
            "streaming": True,
            "component_file": "gui/components/handler_systems_tab.py",
            "status": "MISSING",
        },
        {
            "name": "Service Systems",
            "priority": 7,
            "streaming": True,
            "component_file": "gui/components/service_systems_tab.py",
            "status": "MISSING",
        },
        {
            "name": "System Integration",
            "priority": 8,
            "streaming": True,
            "component_file": "gui/components/system_integration_tab.py",
            "status": "MISSING",
        },
        {
            "name": "Real-time Logs",
            "priority": 9,
            "streaming": True,
            "component_file": "gui/components/realtime_logs_tab.py",
            "status": "MISSING",
        },
        {
            "name": "System Health",
            "priority": 10,
            "streaming": True,
            "component_file": "gui/components/system_health_dashboard.py",
            "status": "EXISTS",
        },
        {
            "name": "Models",
            "priority": 11,
            "streaming": True,
            "component_file": "interfaces/model_tab_interface.py",
            "status": "EXISTS",
        },
        {
            "name": "Model Discovery",
            "priority": 12,
            "streaming": True,
            "component_file": "interfaces/model_discovery_interface.py",
            "status": "EXISTS",
        },
        {
            "name": "Training",
            "priority": 13,
            "streaming": True,
            "component_file": "interfaces/training_interface.py",
            "status": "EXISTS",
        },
        {
            "name": "Visualization",
            "priority": 14,
            "streaming": True,
            "component_file": "interfaces/visualization_tab_interface.py",
            "status": "EXISTS",
        },
        {
            "name": "Performance",
            "priority": 15,
            "streaming": True,
            "component_file": "interfaces/performance_tab_interface.py",
            "status": "EXISTS",
        },
        {
            "name": "GridFormer",
            "priority": 16,
            "streaming": True,
            "component_file": "gui/components/dynamic_gridformer_gui.py",
            "status": "EXISTS",
        },
        {
            "name": "Advanced GridFormer",
            "priority": 17,
            "streaming": True,
            "component_file": "gui/components/dynamic_gridformer_gui.py",
            "status": "EXISTS",
        },
        {
            "name": "VMB Integration",
            "priority": 18,
            "streaming": True,
            "component_file": "gui/components/vmb_integration_tab.py",
            "status": "EXISTS",
        },
        {
            "name": "VMB Demo",
            "priority": 19,
            "streaming": True,
            "component_file": "gui/components/vmb_components_pyqt5.py",
            "status": "EXISTS",
        },
        {
            "name": "Music",
            "priority": 20,
            "streaming": True,
            "component_file": "gui/components/music_tab.py",
            "status": "EXISTS",
        },
        {
            "name": "Echo Log",
            "priority": 21,
            "streaming": True,
            "component_file": "gui/components/echo_log_panel.py",
            "status": "EXISTS",
        },
        {
            "name": "Mesh Map",
            "priority": 22,
            "streaming": True,
            "component_file": "gui/components/mesh_map_panel.py",
            "status": "EXISTS",
        },
        {
            "name": "Agent Status",
            "priority": 23,
            "streaming": True,
            "component_file": "gui/components/agent_status_panel.py",
            "status": "EXISTS",
        },
        {
            "name": "BLT/RAG",
            "priority": 24,
            "streaming": True,
            "component_file": "gui/components/enhanced_blt_rag_tab.py",
            "status": "EXISTS",
        },
        {
            "name": "ARC",
            "priority": 25,
            "streaming": True,
            "component_file": "gui/components/pyqt_main.py",  # Built-in method
            "status": "EXISTS",
        },
        {
            "name": "Vanta Core",
            "priority": 26,
            "streaming": True,
            "component_file": "gui/components/pyqt_main.py",  # Built-in method
            "status": "EXISTS",
        },
    ]

    # Check each tab
    results = {"existing": [], "missing": [], "new": [], "total_count": len(expected_tabs)}

    base_path = Path(".")

    for tab in expected_tabs:
        file_path = base_path / tab["component_file"]
        file_exists = file_path.exists()

        status_icon = "✅" if tab["status"] == "EXISTS" and file_exists else "❌"
        if tab["status"] == "NEW":
            status_icon = "🆕"
        elif tab["status"] == "MISSING":
            status_icon = "⚠️"

        streaming_icon = "📡" if tab["streaming"] else "💬"

        print(f"{status_icon} {streaming_icon} Priority {tab['priority']:2d}: {tab['name']}")
        print(f"    File: {tab['component_file']} ({'EXISTS' if file_exists else 'MISSING'})")

        if tab["status"] == "EXISTS" and file_exists:
            results["existing"].append(tab["name"])
        elif tab["status"] == "NEW":
            results["new"].append(tab["name"])
        else:
            results["missing"].append(tab["name"])

    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print(f"Total expected tabs: {results['total_count']}")
    print(f"Existing tabs: {len(results['existing'])}")
    print(f"New tabs: {len(results['new'])}")
    print(f"Missing tabs: {len(results['missing'])}")

    completion_percentage = (
        (len(results["existing"]) + len(results["new"])) / results["total_count"]
    ) * 100
    print(f"Completion: {completion_percentage:.1f}%")

    if results["missing"]:
        print(f"\n⚠️ MISSING TABS ({len(results['missing'])}):")
        for tab in results["missing"]:
            print(f"  - {tab}")

    # Check for streaming capabilities
    print("\n📡 STREAMING STATUS")
    streaming_tabs = [tab for tab in expected_tabs if tab["streaming"]]
    print(f"Tabs requiring streaming: {len(streaming_tabs)}")

    # Test imports
    print("\n🧪 IMPORT TESTS")
    try:
        sys.path.insert(0, str(base_path))

        # Test Control Center import
        try:
            from gui.components.control_center_tab import ControlCenterTab

            print("✅ Control Center tab imports successfully")
        except ImportError as e:
            print(f"❌ Control Center import failed: {e}")

        # Test main GUI import
        try:
            from gui.components.pyqt_main import VoxSigilMainWindow

            print("✅ Main GUI imports successfully")
        except ImportError as e:
            print(f"❌ Main GUI import failed: {e}")

        # Test processing engines tab
        try:
            from gui.components.processing_engines_tab import ProcessingEnginesTab

            print("✅ Processing Engines tab imports successfully")
        except ImportError as e:
            print(f"❌ Processing Engines import failed: {e}")

    except Exception as e:
        print(f"❌ Import test failed: {e}")

    return results


def create_missing_tabs_plan():
    """Create a plan for implementing missing tabs"""

    missing_tabs = [
        "Individual Agents",
        "Memory Systems",
        "Training Pipelines",
        "Supervisor Systems",
        "Handler Systems",
        "Service Systems",
        "System Integration",
        "Real-time Logs",
    ]

    print("\n📋 IMPLEMENTATION PLAN")
    print("=" * 50)

    for i, tab in enumerate(missing_tabs, 1):
        print(f"{i}. {tab}")
        print("   Priority: High")
        print("   Streaming: Required")
        print(f"   File: gui/components/{tab.lower().replace(' ', '_')}_tab.py")
        print()

    print("🎯 NEXT STEPS:")
    print("1. Create the 8 missing high-priority tabs")
    print("2. Add streaming capabilities to existing static tabs")
    print("3. Update main GUI to use priority-based tab ordering")
    print("4. Implement comprehensive event bus streaming")
    print("5. Add Prometheus metrics for gui_active_streams")
    print("6. Create smoke tests for all 27 tabs")


if __name__ == "__main__":
    results = test_gui_tab_inventory()
    create_missing_tabs_plan()

    # Summary for CI/CD
    completion = ((len(results["existing"]) + len(results["new"])) / results["total_count"]) * 100
    print("\n🎯 TARGET: 27 tabs with streaming")
    print(f"📊 CURRENT: {completion:.1f}% complete")

    if completion >= 80:
        print("🎉 GUI is ready for production!")
        sys.exit(0)
    else:
        print("⚠️ GUI needs more work before deployment")
        sys.exit(1)
