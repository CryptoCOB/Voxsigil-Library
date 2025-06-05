#!/usr/bin/env python3
"""
VMB Import System Check - Final Validation
Tests all VMB system imports and validates functionality
"""

import sys
import os

def test_vmb_imports():
    """Run comprehensive import tests for VMB system"""
    print('🔍 VMB Import System Check - Final Validation')
    print('=' * 60)
    
    # Test 1: Core Python dependencies
    print('\n📦 Testing Core Dependencies:')
    try:
        import yaml
        import asyncio
        import logging
        import dataclasses
        import typing
        from typing import Dict, List, Optional, Any
        print('✅ Core Python modules: SUCCESS')
    except Exception as e:
        print(f'❌ Core Python modules: FAILED - {e}')
        return False
    
    # Test 2: VMB Core System
    print('\n🎯 Testing VMB Core System:')    
    vmb_success = True
    
    try:
        from vmb_activation import CopilotSwarm
        print('✅ VMB Activation: SUCCESS')
    except Exception as e:
        print(f'❌ VMB Activation: FAILED - {e}')
        vmb_success = False
    
    try:
        from vmb_production_executor import ProductionTaskExecutor
        print('✅ VMB Production Executor: SUCCESS')
    except Exception as e:
        print(f'❌ VMB Production Executor: FAILED - {e}')
        vmb_success = False
    
    try:
        import vmb_completion_report
        print('✅ VMB Completion Report: SUCCESS')
    except Exception as e:
        print(f'❌ VMB Completion Report: FAILED - {e}')
        vmb_success = False
    
    # Test 3: Legacy Integration
    print('\n🔗 Testing Legacy Integration:')
    try:
        from voxsigil_supervisor.interfaces.checkin_manager import VantaInteractionManager
        print('✅ VantaInteractionManager: SUCCESS')
    except Exception as e:
        print(f'❌ VantaInteractionManager: FAILED - {e}')
        vmb_success = False
    
    # Test 4: Configuration Loading
    print('\n⚙️ Testing Configuration:')
    try:
        with open('sigil_trace.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print('✅ YAML Configuration Loading: SUCCESS')
        print(f'   Sigil: {config.get("sigil", "Not found")}')
        print(f'   Agent Class: {config.get("agent_class", "Not found")}')
        print(f'   Swarm Variant: {config.get("swarm_variant", "Not found")}')
        print(f'   Role Scope: {config.get("role_scope", "Not found")}')
        print(f'   Activation Mode: {config.get("activation_mode", "Not found")}')
    except Exception as e:
        print(f'❌ YAML Configuration Loading: FAILED - {e}')
        vmb_success = False
      # Test 5: VMB System Instantiation
    print('\n🚀 Testing VMB System Instantiation:')
    try:
        from vmb_activation import CopilotSwarm
        # Load configuration from YAML
        with open('sigil_trace.yaml', 'r') as f:
            config = yaml.safe_load(f)
        swarm = CopilotSwarm(config)
        print('✅ CopilotSwarm instantiation: SUCCESS')
        print(f'   Current agent count: {len(getattr(swarm, "agents", []))}')
        print(f'   Swarm class: {swarm.__class__.__name__}')
    except Exception as e:
        print(f'❌ CopilotSwarm instantiation: FAILED - {e}')
        vmb_success = False
    
    # Test 6: Advanced VMB Components (if available)
    print('\n🔧 Testing Additional VMB Components:')
    additional_tests = [
        ('vmb_advanced_demo', 'VMB Advanced Demo'),
        ('vmb_status', 'VMB Status'),
        ('vmb_operations', 'VMB Operations'),
        ('vmb_final_status', 'VMB Final Status'),
        ('vmb_production_final', 'VMB Production Final')
    ]
    
    for module_name, display_name in additional_tests:
        try:
            __import__(module_name)
            print(f'✅ {display_name}: SUCCESS')
        except ImportError:
            print(f'⚠️ {display_name}: Not available (optional)')
        except Exception as e:
            print(f'❌ {display_name}: FAILED - {e}')
    
    print('\n' + '=' * 60)
    if vmb_success:
        print('🎉 VMB Import System Check COMPLETE - ALL CORE SYSTEMS OPERATIONAL')
        print('🚀 VMB System is ready for full activation!')
    else:
        print('⚠️ VMB Import System Check COMPLETE - Some issues detected')
        print('🔧 Please review failed imports above')
    
    return vmb_success

if __name__ == "__main__":
    success = test_vmb_imports()
    sys.exit(0 if success else 1)
