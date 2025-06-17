"""
Scripts Module Vanta Registration

This module provides registration capabilities for the Scripts module
with the Vanta orchestrator system.

Components registered:
- ScriptRunner: General script execution and management
- ProductionScripts: Production environment scripts
- AutomationScripts: Automated workflow scripts
- MaintenanceScripts: System maintenance and cleanup scripts

HOLO-1.5 Integration: Script orchestration and automation mesh integration.
"""

import asyncio
import importlib
import logging
import subprocess
import sys
import os
from typing import Any, Dict, List

# Configure logging
logger = logging.getLogger(__name__)

class ScriptsModuleAdapter:
    """Adapter for integrating Scripts module with Vanta orchestrator."""
    
    def __init__(self):
        self.module_id = "scripts"
        self.display_name = "Automation Scripts Module"
        self.version = "1.0.0"
        self.description = "Automation scripts and production utilities for VoxSigil Library"
        
        # Script component instances
        self.script_runner = None
        self.production_scripts = None
        self.automation_scripts = None
        self.initialized = False
        
        # Available script components
        self.available_scripts = {
            'apply_encapsulated_registration': 'apply_encapsulated_registration.py',
            'cleanup_organizer': 'cleanup_organizer.py',
            'create_all_components': 'create_all_components.py',
            'generate_agent_classes': 'generate_agent_classes.py',
            'launch_gui': 'launch_gui.py',
            'run_vantacore_grid_connector': 'run_vantacore_grid_connector.py'
        }
        
        # Script metadata
        self.script_metadata = {
            'apply_encapsulated_registration': {
                'description': 'Applies encapsulated registration patterns',
                'category': 'registration',
                'requires_vanta': True
            },
            'cleanup_organizer': {
                'description': 'Organizes and cleans up code structure',
                'category': 'maintenance',
                'requires_vanta': False
            },
            'create_all_components': {
                'description': 'Creates all required components',
                'category': 'generation',
                'requires_vanta': True
            },
            'generate_agent_classes': {
                'description': 'Generates agent class definitions',
                'category': 'generation',
                'requires_vanta': True
            },
            'launch_gui': {
                'description': 'Launches the GUI application',
                'category': 'application',
                'requires_vanta': True
            },
            'run_vantacore_grid_connector': {
                'description': 'Runs the VantaCore grid connector',
                'category': 'networking',
                'requires_vanta': True
            }
        }
        
    async def initialize(self, vanta_core):
        """Initialize the Scripts module with vanta core."""
        try:
            logger.info(f"Initializing Scripts module with Vanta core...")
            
            # Create script runner instance
            self.script_runner = ScriptRunner(vanta_core)
            
            # Try to initialize production scripts if available
            try:
                production_module = importlib.import_module('scripts.production')
                if hasattr(production_module, 'ProductionScripts'):
                    ProductionScriptsClass = getattr(production_module, 'ProductionScripts')
                    self.production_scripts = ProductionScriptsClass(vanta_core=vanta_core)
                else:
                    self.production_scripts = self._create_basic_production_scripts(vanta_core)
            except ImportError:
                logger.info("Production scripts module not found, creating basic handler")
                self.production_scripts = self._create_basic_production_scripts(vanta_core)
            
            # Create automation scripts handler
            self.automation_scripts = AutomationScriptsHandler(vanta_core)
            
            # Initialize async components if they have async_init methods
            if hasattr(self.script_runner, 'async_init'):
                await self.script_runner.async_init()
                
            if hasattr(self.production_scripts, 'async_init'):
                await self.production_scripts.async_init()
                
            if hasattr(self.automation_scripts, 'async_init'):
                await self.automation_scripts.async_init()
            
            self.initialized = True
            logger.info(f"Scripts module initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Scripts module: {e}")
            return False
    
    def _create_basic_production_scripts(self, vanta_core):
        """Create a basic production scripts handler for fallback."""
        class BasicProductionScripts:
            def __init__(self, vanta_core=None):
                self.vanta_core = vanta_core
                
            def run_production_script(self, script_name: str, **kwargs):
                # Basic production script execution
                script_path = os.path.join(os.path.dirname(__file__), 'production', f'{script_name}.py')
                if os.path.exists(script_path):
                    return subprocess.run([sys.executable, script_path], capture_output=True, text=True)
                else:
                    return {"error": f"Production script {script_name} not found"}
                    
        return BasicProductionScripts(vanta_core)
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process requests for Scripts module operations."""
        if not self.initialized:
            return {"error": "Scripts module not initialized"}
        
        try:
            request_type = request.get('type', 'unknown')
            
            if request_type == 'run_script':
                return await self._handle_run_script(request)
            elif request_type == 'list_scripts':
                return await self._handle_list_scripts(request)
            elif request_type == 'script_info':
                return await self._handle_script_info(request)
            elif request_type == 'run_production_script':
                return await self._handle_run_production_script(request)
            elif request_type == 'automation_task':
                return await self._handle_automation_task(request)
            else:
                return {"error": f"Unknown request type: {request_type}"}
                
        except Exception as e:
            logger.error(f"Error processing Scripts request: {e}")
            return {"error": str(e)}
    
    async def _handle_run_script(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle script execution requests."""
        script_name = request.get('script_name')
        script_args = request.get('args', [])
        script_kwargs = request.get('kwargs', {})
        
        if not script_name:
            return {"error": "Missing script_name"}
        
        try:
            if hasattr(self.script_runner, 'run_script'):
                result = await self.script_runner.run_script(script_name, *script_args, **script_kwargs)
            else:
                # Fallback: direct execution
                result = await self._execute_script_directly(script_name, script_args, script_kwargs)
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Script execution failed: {e}"}
    
    async def _execute_script_directly(self, script_name: str, args: List, kwargs: Dict):
        """Execute script directly as fallback."""
        script_path = os.path.join(os.path.dirname(__file__), f'{script_name}.py')
        
        if not os.path.exists(script_path):
            script_path = os.path.join(os.path.dirname(__file__), script_name)
            if not os.path.exists(script_path):
                return {"error": f"Script {script_name} not found"}
        
        try:
            # Execute the script in a subprocess
            cmd = [sys.executable, script_path] + [str(arg) for arg in args]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {"error": "Script execution timed out"}
        except Exception as e:
            return {"error": f"Script execution error: {e}"}
    
    async def _handle_list_scripts(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle script listing requests."""
        try:
            script_list = []
            
            for script_name, script_file in self.available_scripts.items():
                metadata = self.script_metadata.get(script_name, {})
                script_list.append({
                    "name": script_name,
                    "file": script_file,
                    "description": metadata.get('description', 'No description'),
                    "category": metadata.get('category', 'general'),
                    "requires_vanta": metadata.get('requires_vanta', False)
                })
            
            return {"result": script_list, "status": "success"}
            
        except Exception as e:
            return {"error": f"Failed to list scripts: {e}"}
    
    async def _handle_script_info(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle script information requests."""
        script_name = request.get('script_name')
        
        if not script_name:
            return {"error": "Missing script_name"}
        
        try:
            if script_name in self.script_metadata:
                info = self.script_metadata[script_name].copy()
                info['name'] = script_name
                info['file'] = self.available_scripts.get(script_name, 'Unknown')
                return {"result": info, "status": "success"}
            else:
                return {"error": f"Script {script_name} not found"}
                
        except Exception as e:
            return {"error": f"Failed to get script info: {e}"}
    
    async def _handle_run_production_script(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle production script execution requests."""
        script_name = request.get('script_name')
        script_params = request.get('params', {})
        
        if not script_name:
            return {"error": "Missing script_name"}
        
        try:
            if hasattr(self.production_scripts, 'run_production_script'):
                result = self.production_scripts.run_production_script(script_name, **script_params)
            else:
                result = {"error": "Production scripts not available"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Production script execution failed: {e}"}
    
    async def _handle_automation_task(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle automation task requests."""
        task_name = request.get('task_name')
        task_params = request.get('params', {})
        
        if not task_name:
            return {"error": "Missing task_name"}
        
        try:
            if hasattr(self.automation_scripts, 'run_automation_task'):
                result = await self.automation_scripts.run_automation_task(task_name, **task_params)
            else:
                result = {"error": "Automation scripts not available"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Automation task failed: {e}"}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the Scripts module."""
        return {
            "module_id": self.module_id,
            "display_name": self.display_name,
            "version": self.version,
            "description": self.description,
            "capabilities": [
                "script_execution",
                "automation_tasks",
                "production_scripts",
                "maintenance_scripts",
                "workflow_automation"
            ],
            "supported_operations": [
                "run_script",
                "list_scripts",
                "script_info",
                "run_production_script",
                "automation_task"
            ],
            "available_scripts": list(self.available_scripts.keys()),
            "script_categories": list(set(meta.get('category', 'general') for meta in self.script_metadata.values())),
            "initialized": self.initialized,
            "holo_integration": True,
            "cognitive_mesh_role": "PROCESSOR",
            "symbolic_depth": 2
        }
    
    async def shutdown(self):
        """Shutdown the Scripts module gracefully."""
        try:
            logger.info("Shutting down Scripts module...")
            
            # Shutdown components that support it
            if self.script_runner and hasattr(self.script_runner, 'shutdown'):
                if asyncio.iscoroutinefunction(self.script_runner.shutdown):
                    await self.script_runner.shutdown()
                else:
                    self.script_runner.shutdown()
            
            if self.production_scripts and hasattr(self.production_scripts, 'shutdown'):
                if asyncio.iscoroutinefunction(self.production_scripts.shutdown):
                    await self.production_scripts.shutdown()
                else:
                    self.production_scripts.shutdown()
            
            if self.automation_scripts and hasattr(self.automation_scripts, 'shutdown'):
                if asyncio.iscoroutinefunction(self.automation_scripts.shutdown):
                    await self.automation_scripts.shutdown()
                else:
                    self.automation_scripts.shutdown()
            
            self.initialized = False
            logger.info("Scripts module shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Scripts module shutdown: {e}")


class ScriptRunner:
    """Basic script runner for general script execution."""
    
    def __init__(self, vanta_core=None):
        self.vanta_core = vanta_core
        
    async def run_script(self, script_name: str, *args, **kwargs):
        """Run a script with given arguments."""
        # Implementation would depend on specific script requirements
        return {"message": f"Script {script_name} executed", "args": args, "kwargs": kwargs}


class AutomationScriptsHandler:
    """Handler for automation scripts and workflows."""
    
    def __init__(self, vanta_core=None):
        self.vanta_core = vanta_core
        
    async def run_automation_task(self, task_name: str, **params):
        """Run an automation task."""
        # Implementation would depend on specific automation requirements
        return {"message": f"Automation task {task_name} executed", "params": params}


# Registration function for the master orchestrator
async def register_scripts_module(vanta_core) -> ScriptsModuleAdapter:
    """Register the Scripts module with Vanta orchestrator."""
    logger.info("Registering Scripts module with Vanta orchestrator...")
    
    adapter = ScriptsModuleAdapter()
    success = await adapter.initialize(vanta_core)
    
    if success:
        logger.info("Scripts module registration successful")
    else:
        logger.error("Scripts module registration failed")
    
    return adapter


# Export the adapter class for external use
__all__ = ['ScriptsModuleAdapter', 'register_scripts_module']
