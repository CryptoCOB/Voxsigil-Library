#!/usr/bin/env python3
"""
VoxSigil Optimized GUI Validation Test
Test all the optimizations and features in the enhanced GUI.
"""

import sys
import os
import logging
import tracemalloc
from typing import List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("🧪 Testing imports...")
    
    try:
        # Test Qt imports
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
        from PyQt5.QtGui import QPixmap, QKeySequence
        logger.info("✅ PyQt5 imports successful")
        
        # Test optimized GUI import
        import optimized_enhanced_gui
        logger.info("✅ Optimized GUI module imported")
        
        # Test launcher import
        from launch_optimized_gui import main as launcher_main
        logger.info("✅ Launcher module imported")
        
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_data_provider():
    """Test data provider availability"""
    logger.info("🧪 Testing data provider...")
    
    try:
        from gui.components.real_time_data_provider import RealTimeDataProvider
        provider = RealTimeDataProvider()
        metrics = provider.get_all_metrics()
        logger.info(f"✅ Data provider works ({len(metrics)} metrics)")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Data provider issue: {e}")
        return False

def test_gui_styles():
    """Test GUI styles availability"""
    logger.info("🧪 Testing GUI styles...")
    
    try:
        from gui.components.gui_styles import VoxSigilStyles
        logger.info("✅ GUI styles available")
        return True
    except Exception as e:
        logger.warning(f"⚠️ GUI styles issue: {e}")
        return False

def test_psutil():
    """Test psutil availability for resource monitoring"""
    logger.info("🧪 Testing resource monitoring...")
    
    try:
        import psutil
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        logger.info(f"✅ Resource monitoring available (CPU: {cpu:.1f}%, RAM: {mem:.1f}%)")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Resource monitoring issue: {e}")
        return False

def test_gui_creation():
    """Test GUI creation without display"""
    logger.info("🧪 Testing GUI creation...")
    
    try:
        # Start memory tracking
        tracemalloc.start()
        
        # Import and create app (no display)
        from PyQt5.QtWidgets import QApplication
        from optimized_enhanced_gui import OptimizedEnhancedGUI
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Create GUI instance
        gui = OptimizedEnhancedGUI()
        
        # Test basic properties
        assert gui.windowTitle() == "VoxSigil Enhanced GUI - Optimized"
        assert gui.tabs.count() >= 6  # Should have at least 6 tabs
        
        # Test tab names
        tab_titles = [gui.tabs.tabText(i) for i in range(gui.tabs.count())]
        expected_tabs = ["📊 Status", "📡 Dashboard", "🤖 Models", "🎯 Training", "📈 Visualization", "🎵 Music", "💓 Heartbeat"]
        
        for expected in expected_tabs:
            if not any(expected in title for title in tab_titles):
                logger.warning(f"⚠️ Missing expected tab: {expected}")
        
        logger.info(f"✅ GUI created successfully with {gui.tabs.count()} tabs")
        
        # Check memory usage
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            logger.info(f"📊 Memory usage: Current {current/1024/1024:.1f}MB, Peak {peak/1024/1024:.1f}MB")
            tracemalloc.stop()
        
        # Clean up
        gui.close()
        del gui
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GUI creation failed: {e}")
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        return False

def test_optimization_features():
    """Test optimization feature availability"""
    logger.info("🧪 Testing optimization features...")
    
    try:
        from optimized_enhanced_gui import OptimizedLazyTab, TabLoadWorker, FeatureSpec
        
        # Test feature spec
        spec = FeatureSpec("Test", lambda: None, 5000, 2)
        assert spec.title == "Test"
        assert spec.timeout_ms == 5000
        assert spec.max_retries == 2
        
        logger.info("✅ Optimization classes available")
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization features failed: {e}")
        return False

def run_validation():
    """Run complete validation suite"""
    logger.info("🚀 VoxSigil Optimized GUI Validation")
    logger.info("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Data Provider", test_data_provider),
        ("GUI Styles", test_gui_styles),
        ("Resource Monitoring", test_psutil),
        ("Optimization Features", test_optimization_features),
        ("GUI Creation", test_gui_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.warning(f"⚠️ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 60)
    logger.info(f"📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Optimized GUI is ready!")
        logger.info("💡 You can now launch: python launch_optimized_gui.py")
        logger.info("💡 Or use the batch file: Launch_Optimized_Enhanced_GUI.bat")
        return True
    else:
        logger.warning(f"⚠️ {total-passed} tests failed - check issues above")
        return False

if __name__ == "__main__":
    try:
        success = run_validation()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Validation crashed: {e}")
        sys.exit(1)
