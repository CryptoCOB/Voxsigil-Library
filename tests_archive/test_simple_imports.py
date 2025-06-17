#!/usr/bin/env python3
"""
Simple import test for enhanced tabs without emojis.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test if all components import successfully."""
    results = []

    # Test enhanced tabs
    tabs_to_test = [
        ("Enhanced Model Tab", "gui.components.enhanced_model_tab", "EnhancedModelTab"),
        (
            "Enhanced Model Discovery Tab",
            "gui.components.enhanced_model_discovery_tab",
            "EnhancedModelDiscoveryTab",
        ),
        (
            "Enhanced Visualization Tab",
            "gui.components.enhanced_visualization_tab",
            "EnhancedVisualizationTab",
        ),
        ("Enhanced Music Tab", "gui.components.enhanced_music_tab", "EnhancedMusicTab"),
        ("Enhanced Training Tab", "gui.components.enhanced_training_tab", "EnhancedTrainingTab"),
        ("Dev Mode Panel", "gui.components.dev_mode_panel", "DevModeControlPanel"),
        (
            "Music Composer Agent",
            "agents.ensemble.music.music_composer_agent",
            "MusicComposerAgent",
        ),
    ]

    for name, module_name, class_name in tabs_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            results.append((name, True, None))
            print(f"OK: {name}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"FAIL: {name} - {e}")

    # Summary
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"\nResults: {passed}/{total} components imported successfully")

    if passed == total:
        print("SUCCESS: All components are working")
        return True
    else:
        print("PARTIAL: Some components have issues")
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
