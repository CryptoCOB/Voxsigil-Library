#!/usr/bin/env python
"""
Pattern Detection Test Script

This script demonstrates the enhanced pattern detection capabilities.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run pattern detection tests."""
    # Add the project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        # Import pattern detection
        from MetaConsciousness.utils.pattern_detection import (
            detect_checkerboard,
            detect_gradient,
            detect_adversarial_patterns,
            detect_repeating_patterns,
            detect_noise_level,
            detect_edges,
            detect_cycles,
            analyze_patterns
        )
        
        print("\n🔍 TESTING ENHANCED PATTERN DETECTION")
        print("======================================")
        
        # Test 1: Checkerboard pattern
        print("\n🧩 Testing Checkerboard Detection")
        checkerboard = np.zeros((20, 20))
        checkerboard[::2, ::2] = 1
        checkerboard[1::2, 1::2] = 1
        
        # Test detection with 2 scales
        for scale in [1, 2]:
            result = detect_checkerboard(checkerboard, scale=scale)
            print(f"Scale {scale}: {'✅ Detected' if result['detected'] else '❌ Not detected'} with confidence {result.get('confidence', 0):.3f}")
        
        # Test 2: Gradient pattern
        print("\n📈 Testing Gradient Detection")
        gradient = np.linspace(0, 1, 20)
        gradient_2d = np.tile(gradient, (20, 1))
        
        # Test both methods (numpy and sobel)
        for method in ['numpy', 'sobel']:
            result = detect_gradient(gradient_2d, axis=0)
            print(f"Method {method}: {'✅ Detected' if result['detected'] else '❌ Not detected'} with confidence {result.get('confidence', 0):.3f}")
        
        # Test 3: Noise level detection
        print("\n🔊 Testing Noise Level Detection")
        noise = np.random.normal(0, 1, (20, 20))
        for method in ['std_dev', 'entropy', 'high_freq']:
            result = detect_noise_level(noise, method=method)
            print(f"Method {method}: Noise level = {result.get('confidence', 0):.3f} ({'✅ High' if result['detected'] else '❌ Low'})")
        
        # Test 4: Edge detection
        print("\n🧱 Testing Edge Detection")
        edges = np.zeros((20, 20))
        edges[5:15, 5:15] = 1  # Create a square
        
        result = detect_edges(edges)
        print(f"Edge detection: {'✅ Detected' if result['detected'] else '❌ Not detected'} with density {result.get('edge_density', 0):.3f}")
        
        # Test 5: Cycle detection
        print("\n🔄 Testing Cycle Detection")
        # Create a sinusoidal pattern with period 10
        x = np.arange(100)
        sinusoidal = np.sin(2 * np.pi * x / 10) + 0.1 * np.random.randn(100)
        
        result = detect_cycles(sinusoidal, min_period=5, max_period=20)
        if result['detected']:
            print(f"✅ Cycle detected with period {result.get('detected_period')} (confidence: {result.get('confidence', 0):.3f})")
        else:
            print(f"❌ No cycles detected: {result.get('reason', 'Unknown reason')}")
        
        # Test 6: Repeating patterns
        print("\n🔁 Testing Repeating Pattern Detection")
        # Create a repeating pattern
        pattern = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        
        for method in ['autocorr', 'fft_peak']:
            result = detect_repeating_patterns(np.array(pattern), method=method)
            print(f"Method {method}: {'✅ Detected' if result['detected'] else '❌ Not detected'} with confidence {result.get('confidence', 0):.3f}")
        
        # Test 7: Comprehensive analysis
        print("\n🔬 Testing Comprehensive Analysis")
        # Create a mixture of patterns
        mixed = np.zeros((30, 30))
        # Add checkerboard to one quarter
        mixed[:15, :15] = np.tile(np.array([[0, 1], [1, 0]]), (8, 8))[:15, :15]
        # Add gradient to another quarter
        x, y = np.indices((15, 15))
        mixed[15:, 15:] = (x[:15] + y[:15]) / 30
        
        # Run analysis
        analysis = analyze_patterns(mixed)
        
        # Print detected patterns
        print("Detected patterns:")
        for pattern_name, result in analysis.items():
            if isinstance(result, dict) and result.get('detected', False):
                confidence = result.get('confidence', 0)
                print(f"- {pattern_name}: ✅ Detected with confidence {confidence:.3f}")
        
        # Print dominant pattern
        if 'dominant_pattern' in analysis and analysis['dominant_pattern'] != 'none':
            print(f"\nDominant pattern: {analysis['dominant_pattern']} (confidence: {analysis.get('dominant_confidence', 0):.3f})")
        else:
            print("\nNo dominant pattern detected")
        
        print("\n✅ Pattern detection tests complete.")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
