# Vanta/registration/__init__.py
"""
Vanta Registration System
=========================

Provides centralized registration infrastructure for all VoxSigil Library modules.

Key Components:
- MasterRegistrationOrchestrator - Coordinates all module registrations
- Individual module registration helpers
- Registration status tracking and reporting
- Integration with Vanta orchestrator system

Usage:
    from Vanta.registration import register_all_modules, get_registration_status
    
    # Register all modules
    results = await register_all_modules()
    
    # Check status
    status = get_registration_status()
"""

from .master_registration import (
    register_all_modules,
    get_registration_status,
    registration_orchestrator,
    RegistrationOrchestrator
)

__all__ = [
    'register_all_modules',
    'get_registration_status', 
    'registration_orchestrator',
    'RegistrationOrchestrator'
]
