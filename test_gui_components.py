#!/usr/bin/env python3
"""
Test script to check which GUI components can be imported and work correctly.
This will show which tabs will use real components vs fallback tabs.
"""

import logging
import sys
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_component_import(component_path, component_name):
    """Test if a component can be imported successfully"""
    try:
        module = __import__(component_path, fromlist=[component_name])
        component_class = getattr(module, component_name)
        logger.info(f"✅ {component_name} from {component_path} - SUCCESS")
        return True, component_class
    except ImportError as e:
        logger.warning(f"❌ {component_name} from {component_path} - Import Error: {e}")
        return False, None
    except AttributeError as e:
        logger.warning(f"❌ {component_name} from {component_path} - Attribute Error: {e}")
        return False, None
    except Exception as e:
        logger.error(f"❌ {component_name} from {component_path} - Unexpected Error: {e}")
        return False, None

def test_component_instantiation(component_class, component_name):
    """Test if a component can be instantiated successfully"""
    try:
        # Try to instantiate without Qt application (just for testing)
        instance = component_class()
        logger.info(f"✅ {component_name} instantiation - SUCCESS")
        return True
    except Exception as e:
        logger.warning(f"⚠️ {component_name} instantiation - Error: {e}")
        return False

def main():
    """Test all GUI components used in the complete GUI"""
    
    print("🔍 Testing VoxSigil GUI Component Imports\n")
    
    # Components to test (as used in complete_live_gui.py)
    components_to_test = [
        # Core system components
        ("gui.components.mesh_map_panel", "MeshMapPanel"),
        ("gui.components.vanta_core_tab", "VantaCoreTab"),
        ("gui.components.streaming_dashboard", "StreamingDashboard"),
        
        # Agent management components
        ("gui.components.individual_agents_tab", "IndividualAgentsTab"),
        ("gui.components.enhanced_agent_status_panel", "EnhancedAgentStatusPanel"),
        ("gui.components.enhanced_agent_status_panel_v2", "EnhancedAgentStatusPanelV2"),
        ("gui.components.agent_status_panel", "AgentStatusPanel"),
        ("gui.components.heartbeat_monitor_tab", "HeartbeatMonitorTab"),
        
        # Training and ML components
        ("gui.components.training_control_tab", "TrainingControlTab"),
        ("gui.components.enhanced_training_tab", "EnhancedTrainingTab"),
        ("gui.components.enhanced_model_tab", "EnhancedModelTab"),
        ("gui.components.enhanced_model_discovery_tab", "EnhancedModelDiscoveryTab"),
        ("gui.components.dataset_panel", "DatasetPanel"),
        ("gui.components.experiment_tracker_tab", "ExperimentTrackerTab"),
        ("gui.components.enhanced_gridformer_tab", "EnhancedGridFormerTab"),
        ("gui.components.dynamic_gridformer_gui", "DynamicGridFormerGUI"),
        
        # Monitoring and analytics components
        ("gui.components.system_health_dashboard", "SystemHealthDashboard"),
        ("gui.components.heartbeat_monitor", "HeartbeatMonitor"),
        ("gui.components.enhanced_visualization_tab", "EnhancedVisualizationTab"),
        ("gui.components.notification_center_tab", "NotificationCenterTab"),
        ("gui.components.realtime_logs_tab", "RealtimeLogsTab"),
        
        # Audio and media components
        ("gui.components.enhanced_neural_tts_tab", "EnhancedNeuralTTSTab"),
        ("gui.components.enhanced_music_tab", "EnhancedMusicTab"),
        ("gui.components.music_tab", "MusicGenerationTab"),
        
        # Development and system components
        ("gui.components.memory_systems_tab", "MemorySystemsTab"),
        ("gui.components.dev_mode_panel", "DevModePanel"),
        ("gui.components.enhanced_echo_log_panel", "EnhancedEchoLogPanel"),
        ("gui.components.config_editor_tab", "ConfigEditorTab"),
        ("gui.components.security_panel", "SecurityPanel"),
        ("gui.components.dependency_panel", "DependencyPanel"),
        ("gui.components.system_integration_tab", "SystemIntegrationTab"),
        ("gui.components.enhanced_blt_rag_tab", "EnhancedBLTRAGTab"),
        
        # Specialized components
        ("gui.components.processing_engines_tab", "ProcessingEnginesTab"),
        ("gui.components.supervisor_systems_tab", "SupervisorSystemsTab"),
        ("gui.components.novel_reasoning_tab", "NovelReasoningTab"),
    ]
    
    successful_imports = []
    failed_imports = []
    instantiation_successes = []
    instantiation_failures = []
    
    print("=" * 60)
    print("IMPORT TESTING")
    print("=" * 60)
    
    for component_path, component_name in components_to_test:
        success, component_class = test_component_import(component_path, component_name)
        if success:
            successful_imports.append((component_path, component_name))
            # Test instantiation (without PyQt5 app)
            if test_component_instantiation(component_class, component_name):
                instantiation_successes.append((component_path, component_name))
            else:
                instantiation_failures.append((component_path, component_name))
        else:
            failed_imports.append((component_path, component_name))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ SUCCESSFUL IMPORTS ({len(successful_imports)}):")
    for path, name in successful_imports:
        print(f"   • {name} ({path})")
    
    print(f"\n❌ FAILED IMPORTS ({len(failed_imports)}):")
    for path, name in failed_imports:
        print(f"   • {name} ({path})")
    
    print(f"\n🟢 SUCCESSFUL INSTANTIATIONS ({len(instantiation_successes)}):")
    for path, name in instantiation_successes:
        print(f"   • {name}")
    
    print(f"\n🟡 FAILED INSTANTIATIONS ({len(instantiation_failures)}):")
    for path, name in instantiation_failures:
        print(f"   • {name}")
    
    # Calculate percentages
    total_components = len(components_to_test)
    import_success_rate = (len(successful_imports) / total_components) * 100
    instantiation_success_rate = (len(instantiation_successes) / total_components) * 100
    
    print("\n📊 STATISTICS:")
    print(f"   Total Components Tested: {total_components}")
    print(f"   Import Success Rate: {import_success_rate:.1f}%")
    print(f"   Instantiation Success Rate: {instantiation_success_rate:.1f}%")
    
    if import_success_rate > 50:
        print(f"\n🎉 Good news! {import_success_rate:.1f}% of components can be imported.")
        print("   The GUI should show many real interactive components!")
    else:
        print(f"\n⚠️ Warning: Only {import_success_rate:.1f}% of components can be imported.")
        print("   Most tabs will show fallback interfaces.")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
