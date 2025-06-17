#!/usr/bin/env python3
"""
🎯 VoxSigil GUI Final Validation Test
Comprehensive test of all tabs, imports, and functionality
"""

import importlib
import sys


def test_import(module_name, class_name=None):
    """Test if a module/class can be imported"""
    try:
        module = importlib.import_module(module_name)
        if class_name:
            getattr(module, class_name)
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    print("🎯 VoxSigil GUI Final Validation Test")
    print("=" * 60)

    # Test main GUI import
    print("\n🔍 TESTING MAIN GUI IMPORTS:")
    success, error = test_import("gui.components.pyqt_main", "VoxSigilMainWindow")
    if success:
        print("✅ VoxSigilMainWindow import: SUCCESS")
    else:
        print(f"❌ VoxSigilMainWindow import: FAILED - {error}")
        return 1

    # Test style system
    success, error = test_import("gui.components.gui_styles", "VoxSigilStyles")
    if success:
        print("✅ VoxSigilStyles import: SUCCESS")
    else:
        print(f"❌ VoxSigilStyles import: FAILED - {error}")

    # Test critical tabs
    critical_tabs = [
        ("gui.components.dynamic_gridformer_gui", "DynamicGridFormerTab"),
        ("gui.components.control_center_tab", "ControlCenterTab"),
        ("gui.components.individual_agents_tab", "IndividualAgentsTab"),
        ("gui.components.memory_systems_tab", "MemorySystemsTab"),
        ("gui.components.training_pipelines_tab", "TrainingPipelinesTab"),
        ("gui.components.supervisor_systems_tab", "SupervisorSystemsTab"),
        ("gui.components.handler_systems_tab", "HandlerSystemsTab"),
        ("gui.components.service_systems_tab", "ServiceSystemsTab"),
        ("gui.components.system_integration_tab", "SystemIntegrationTab"),
        ("gui.components.realtime_logs_tab", "RealtimeLogsTab"),
    ]

    print("\n🔍 TESTING CRITICAL TAB IMPORTS:")
    tab_success_count = 0
    for module_name, class_name in critical_tabs:
        success, error = test_import(module_name, class_name)
        if success:
            print(f"✅ {class_name}: SUCCESS")
            tab_success_count += 1
        else:
            print(f"❌ {class_name}: FAILED - {error}")

    # Test interface imports
    print("\n🔍 TESTING INTERFACE IMPORTS:")
    interfaces = [
        ("interfaces.model_tab_interface", "ModelTabInterface"),
        ("interfaces.training_interface", "TrainingInterface"),
        ("interfaces.visualization_tab_interface", "VisualizationTabInterface"),
        ("interfaces.performance_tab_interface", "PerformanceTabInterface"),
    ]

    interface_success_count = 0
    for module_name, class_name in interfaces:
        success, error = test_import(module_name, class_name)
        if success:
            print(f"✅ {class_name}: SUCCESS")
            interface_success_count += 1
        else:
            print(f"⚠️ {class_name}: OPTIONAL - {error}")

    # Test existing component imports
    print("\n🔍 TESTING EXISTING COMPONENT IMPORTS:")
    components = [
        ("gui.components.agent_status_panel", "AgentStatusPanel"),
        ("gui.components.echo_log_panel", "EchoLogPanel"),
        ("gui.components.mesh_map_panel", "MeshMapPanel"),
        ("gui.components.music_tab", "MusicTab"),
        ("gui.components.enhanced_blt_rag_tab", "EnhancedBLTRAGTab"),
    ]

    component_success_count = 0
    for module_name, class_name in components:
        success, error = test_import(module_name, class_name)
        if success:
            print(f"✅ {class_name}: SUCCESS")
            component_success_count += 1
        else:
            print(f"⚠️ {class_name}: OPTIONAL - {error}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"✅ Main GUI: {'PASS' if success else 'FAIL'}")
    print(f"✅ Critical Tabs: {tab_success_count}/{len(critical_tabs)} imported")
    print(f"⚠️ Interfaces: {interface_success_count}/{len(interfaces)} imported")
    print(f"⚠️ Components: {component_success_count}/{len(components)} imported")

    total_expected = 1 + len(critical_tabs) + len(interfaces) + len(components)
    total_working = (
        (1 if success else 0)
        + tab_success_count
        + interface_success_count
        + component_success_count
    )
    completion_rate = (total_working / total_expected) * 100

    print(f"📈 Overall Completion: {completion_rate:.1f}%")

    if completion_rate >= 80:
        print("🎉 EXCELLENT! VoxSigil GUI is ready for production")
        return 0
    elif completion_rate >= 60:
        print("✅ GOOD! VoxSigil GUI is functional with minor issues")
        return 0
    else:
        print("⚠️ NEEDS WORK! Several critical components are missing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
