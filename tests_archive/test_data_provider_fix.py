#!/usr/bin/env python3
"""
Test the RealTimeDataProvider fix
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test importing RealTimeDataProvider"""
    try:
        from gui.components.real_time_data_provider import RealTimeDataProvider
        print("✅ RealTimeDataProvider import successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_instantiation():
    """Test creating RealTimeDataProvider instance"""
    try:
        from gui.components.real_time_data_provider import RealTimeDataProvider
        provider = RealTimeDataProvider()
        print("✅ RealTimeDataProvider instantiation successful")
        return provider
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        return None

def test_methods(provider):
    """Test provider methods"""
    if not provider:
        return False
        
    try:
        # Test getting all metrics
        metrics = provider.get_all_metrics()
        print(f"✅ get_all_metrics() returned {len(metrics)} metrics")
        
        # Test individual metric sources
        system_metrics = provider.get_system_metrics()
        print(f"✅ get_system_metrics() returned {len(system_metrics)} metrics")
        
        vanta_metrics = provider.get_vanta_metrics()
        print(f"✅ get_vanta_metrics() returned {len(vanta_metrics)} metrics")
        
        training_metrics = provider.get_training_metrics()
        print(f"✅ get_training_metrics() returned {len(training_metrics)} metrics")
        
        audio_metrics = provider.get_audio_metrics()
        print(f"✅ get_audio_metrics() returned {len(audio_metrics)} metrics")
        
        return True
    except Exception as e:
        print(f"❌ Method testing failed: {e}")
        return False

def main():
    print("🔧 Testing RealTimeDataProvider Fix")
    print("=" * 50)
    
    # Test import
    if not test_import():
        return False
    
    # Test instantiation
    provider = test_instantiation()
    if not provider:
        return False
    
    # Test methods
    if not test_methods(provider):
        return False
    
    print("=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("✅ RealTimeDataProvider is working correctly")
    print("✅ Ready for GUI launch!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 You can now run: python launch_enhanced_gui_clean.py")
    else:
        print("\n❌ Fix needed before launching GUI")
