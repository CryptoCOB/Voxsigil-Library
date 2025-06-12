"""
Strategies Module Vanta Registration

This module provides registration capabilities for the Strategies module
with the Vanta orchestrator system.

Components registered:
- ExecutionStrategy: Base execution strategy interface and implementations
- EvaluationHeuristics: Evaluation and scoring heuristics
- RetryPolicy: Retry and fallback policy implementations
- ScaffoldRouter: Reasoning scaffold routing logic

HOLO-1.5 Integration: Strategic decision-making and execution orchestration capabilities.
"""

import asyncio
import importlib
import logging
from typing import Any, Dict, List, Optional, Type, Union

# Configure logging
logger = logging.getLogger(__name__)

class StrategiesModuleAdapter:
    """Adapter for integrating Strategies module with Vanta orchestrator."""
    
    def __init__(self):
        self.module_id = "strategies"
        self.display_name = "Execution Strategies Module"
        self.version = "1.0.0"
        self.description = "Strategic execution patterns and decision-making logic for VoxSigil operations"
        
        # Strategy instances
        self.execution_strategies = {}
        self.evaluation_heuristics = None
        self.retry_policy = None
        self.scaffold_router = None
        self.initialized = False
        
        # Available strategy components
        self.available_strategies = {
            'execution_strategy': 'execution_strategy.BaseExecutionStrategy',
            'evaluation_heuristics': 'evaluation_heuristics.EvaluationHeuristics',
            'retry_policy': 'retry_policy.RetryPolicy',
            'scaffold_router': 'scaffold_router.ScaffoldRouter'
        }
        
    async def initialize(self, vanta_core):
        """Initialize the Strategies module with vanta core."""
        try:
            logger.info(f"Initializing Strategies module with Vanta core...")
            
            # Initialize Execution Strategy
            try:
                execution_strategy_module = importlib.import_module('strategies.execution_strategy')
                
                # Look for available execution strategy classes
                strategy_classes = []
                for attr_name in dir(execution_strategy_module):
                    attr = getattr(execution_strategy_module, attr_name)
                    if (isinstance(attr, type) and 
                        hasattr(attr, 'execute_scaffold') and 
                        attr_name != 'BaseExecutionStrategy'):
                        strategy_classes.append((attr_name, attr))
                
                # Initialize available strategies
                for strategy_name, strategy_class in strategy_classes:
                    try:
                        strategy_instance = self._initialize_component(strategy_class, vanta_core)
                        if strategy_instance:
                            self.execution_strategies[strategy_name] = strategy_instance
                            logger.info(f"Initialized execution strategy: {strategy_name}")
                    except Exception as e:
                        logger.warning(f"Failed to initialize strategy {strategy_name}: {e}")
                
                # If no concrete strategies found, store the base class for reference
                if not self.execution_strategies and hasattr(execution_strategy_module, 'BaseExecutionStrategy'):
                    self.execution_strategies['base'] = execution_strategy_module.BaseExecutionStrategy
                    
            except ImportError as e:
                logger.warning(f"Execution Strategy not available: {e}")
            
            # Initialize Evaluation Heuristics
            try:
                eval_heuristics_module = importlib.import_module('strategies.evaluation_heuristics')
                
                # Look for heuristics classes
                for attr_name in dir(eval_heuristics_module):
                    attr = getattr(eval_heuristics_module, attr_name)
                    if isinstance(attr, type) and 'heuristic' in attr_name.lower():
                        self.evaluation_heuristics = self._initialize_component(attr, vanta_core)
                        break
                
                # Fallback to module-level functions
                if not self.evaluation_heuristics:
                    self.evaluation_heuristics = eval_heuristics_module
                    
            except ImportError as e:
                logger.warning(f"Evaluation Heuristics not available: {e}")
            
            # Initialize Retry Policy
            try:
                retry_policy_module = importlib.import_module('strategies.retry_policy')
                
                # Look for policy classes
                for attr_name in dir(retry_policy_module):
                    attr = getattr(retry_policy_module, attr_name)
                    if isinstance(attr, type) and 'policy' in attr_name.lower():
                        self.retry_policy = self._initialize_component(attr, vanta_core)
                        break
                
                # Fallback to module-level functions
                if not self.retry_policy:
                    self.retry_policy = retry_policy_module
                    
            except ImportError as e:
                logger.warning(f"Retry Policy not available: {e}")
            
            # Initialize Scaffold Router
            try:
                scaffold_router_module = importlib.import_module('strategies.scaffold_router')
                
                # Look for router classes
                for attr_name in dir(scaffold_router_module):
                    attr = getattr(scaffold_router_module, attr_name)
                    if isinstance(attr, type) and 'router' in attr_name.lower():
                        self.scaffold_router = self._initialize_component(attr, vanta_core)
                        break
                
                # Fallback to module-level functions
                if not self.scaffold_router:
                    self.scaffold_router = scaffold_router_module
                    
            except ImportError as e:
                logger.warning(f"Scaffold Router not available: {e}")
            
            # Initialize async components
            for strategy in self.execution_strategies.values():
                if strategy and hasattr(strategy, 'async_init'):
                    await strategy.async_init()
            
            for component in [self.evaluation_heuristics, self.retry_policy, self.scaffold_router]:
                if component and hasattr(component, 'async_init'):
                    await component.async_init()
            
            self.initialized = True
            logger.info(f"Strategies module initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Strategies module: {e}")
            return False
    
    def _initialize_component(self, component_class, vanta_core):
        """Helper method to initialize a component with vanta_core."""
        try:
            # Try to initialize with vanta_core
            return component_class(vanta_core=vanta_core)
        except TypeError:
            try:
                # Try without vanta_core
                instance = component_class()
                # Try to set vanta_core afterwards
                if hasattr(instance, 'set_vanta_core'):
                    instance.set_vanta_core(vanta_core)
                elif hasattr(instance, 'vanta_core'):
                    instance.vanta_core = vanta_core
                return instance
            except Exception as e:
                logger.warning(f"Failed to initialize component {component_class.__name__}: {e}")
                return None
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process requests for Strategy module operations."""
        if not self.initialized:
            return {"error": "Strategies module not initialized"}
        
        try:
            request_type = request.get('type', 'unknown')
            
            if request_type == 'execute_scaffold':
                return await self._handle_scaffold_execution(request)
            elif request_type == 'evaluate_result':
                return await self._handle_evaluation(request)
            elif request_type == 'retry_operation':
                return await self._handle_retry(request)
            elif request_type == 'route_scaffold':
                return await self._handle_scaffold_routing(request)
            elif request_type == 'strategy_selection':
                return await self._handle_strategy_selection(request)
            else:
                return {"error": f"Unknown request type: {request_type}"}
                
        except Exception as e:
            logger.error(f"Error processing Strategies request: {e}")
            return {"error": str(e)}
    
    async def _handle_scaffold_execution(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle scaffold execution requests."""
        if not self.execution_strategies:
            return {"error": "No execution strategies available"}
        
        query = request.get('query')
        scaffold = request.get('scaffold')
        context = request.get('context', {})
        strategy_name = request.get('strategy', 'default')
        
        # Select strategy
        strategy = None
        if strategy_name in self.execution_strategies:
            strategy = self.execution_strategies[strategy_name]
        elif self.execution_strategies:
            strategy = list(self.execution_strategies.values())[0]
        
        if not strategy:
            return {"error": "No suitable execution strategy found"}
        
        try:
            if hasattr(strategy, 'execute_scaffold'):
                if asyncio.iscoroutinefunction(strategy.execute_scaffold):
                    result = await strategy.execute_scaffold(query, scaffold, context)
                else:
                    result = strategy.execute_scaffold(query, scaffold, context)
            else:
                return {"error": "Strategy does not support scaffold execution"}
            
            return {"result": result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Scaffold execution failed: {e}"}
    
    async def _handle_evaluation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle evaluation requests."""
        if not self.evaluation_heuristics:
            return {"error": "Evaluation heuristics not available"}
        
        result_data = request.get('result_data')
        evaluation_criteria = request.get('criteria', {})
        
        try:
            if hasattr(self.evaluation_heuristics, 'evaluate'):
                if asyncio.iscoroutinefunction(self.evaluation_heuristics.evaluate):
                    evaluation = await self.evaluation_heuristics.evaluate(result_data, evaluation_criteria)
                else:
                    evaluation = self.evaluation_heuristics.evaluate(result_data, evaluation_criteria)
            elif hasattr(self.evaluation_heuristics, 'score'):
                evaluation = self.evaluation_heuristics.score(result_data, **evaluation_criteria)
            else:
                return {"error": "Evaluation method not found"}
            
            return {"evaluation": evaluation, "status": "success"}
            
        except Exception as e:
            return {"error": f"Evaluation failed: {e}"}
    
    async def _handle_retry(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle retry policy requests."""
        if not self.retry_policy:
            return {"error": "Retry policy not available"}
        
        operation = request.get('operation')
        attempt_count = request.get('attempt_count', 1)
        error_info = request.get('error_info', {})
        
        try:
            if hasattr(self.retry_policy, 'should_retry'):
                should_retry = self.retry_policy.should_retry(attempt_count, error_info)
                
                if should_retry:
                    delay = getattr(self.retry_policy, 'get_delay', lambda x: 1.0)(attempt_count)
                    return {
                        "should_retry": True,
                        "delay": delay,
                        "status": "success"
                    }
                else:
                    return {
                        "should_retry": False,
                        "status": "success"
                    }
            else:
                return {"error": "Retry policy method not found"}
            
        except Exception as e:
            return {"error": f"Retry policy evaluation failed: {e}"}
    
    async def _handle_scaffold_routing(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle scaffold routing requests."""
        if not self.scaffold_router:
            return {"error": "Scaffold router not available"}
        
        query = request.get('query')
        available_scaffolds = request.get('available_scaffolds', [])
        context = request.get('context', {})
        
        try:
            if hasattr(self.scaffold_router, 'route_scaffold'):
                if asyncio.iscoroutinefunction(self.scaffold_router.route_scaffold):
                    route_result = await self.scaffold_router.route_scaffold(
                        query, available_scaffolds, context
                    )
                else:
                    route_result = self.scaffold_router.route_scaffold(
                        query, available_scaffolds, context
                    )
            elif hasattr(self.scaffold_router, 'select_scaffold'):
                route_result = self.scaffold_router.select_scaffold(
                    query, available_scaffolds, **context
                )
            else:
                return {"error": "Scaffold routing method not found"}
            
            return {"route_result": route_result, "status": "success"}
            
        except Exception as e:
            return {"error": f"Scaffold routing failed: {e}"}
    
    async def _handle_strategy_selection(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle strategy selection requests."""
        query_type = request.get('query_type')
        complexity = request.get('complexity', 'medium')
        constraints = request.get('constraints', {})
        
        try:
            # Simple strategy selection logic
            available_strategies = list(self.execution_strategies.keys())
            
            if not available_strategies:
                return {"error": "No strategies available"}
            
            # Basic selection logic (can be enhanced)
            if complexity == 'high' and len(available_strategies) > 1:
                selected_strategy = available_strategies[-1]  # Last one might be most advanced
            else:
                selected_strategy = available_strategies[0]  # First available
            
            return {
                "selected_strategy": selected_strategy,
                "available_strategies": available_strategies,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Strategy selection failed: {e}"}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the Strategies module."""
        return {
            "module_id": self.module_id,
            "display_name": self.display_name,
            "version": self.version,
            "description": self.description,
            "capabilities": [
                "scaffold_execution",
                "result_evaluation",
                "retry_policies",
                "scaffold_routing",
                "strategy_selection",
                "execution_orchestration"
            ],
            "supported_operations": [
                "execute_scaffold",
                "evaluate_result",
                "retry_operation",
                "route_scaffold",
                "strategy_selection"
            ],
            "execution_strategies": list(self.execution_strategies.keys()),
            "strategy_components": list(self.available_strategies.keys()),
            "initialized": self.initialized,
            "holo_integration": True,
            "cognitive_mesh_role": "ORCHESTRATOR",
            "symbolic_depth": 4
        }
    
    async def shutdown(self):
        """Shutdown the Strategies module gracefully."""
        try:
            logger.info("Shutting down Strategies module...")
            
            # Shutdown all strategies that support it
            for strategy in self.execution_strategies.values():
                if strategy and hasattr(strategy, 'shutdown'):
                    if asyncio.iscoroutinefunction(strategy.shutdown):
                        await strategy.shutdown()
                    else:
                        strategy.shutdown()
            
            # Shutdown other components
            components = [self.evaluation_heuristics, self.retry_policy, self.scaffold_router]
            for component in components:
                if component and hasattr(component, 'shutdown'):
                    if asyncio.iscoroutinefunction(component.shutdown):
                        await component.shutdown()
                    else:
                        component.shutdown()
            
            self.initialized = False
            logger.info("Strategies module shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Strategies module shutdown: {e}")


# Registration function for the master orchestrator
async def register_strategies_module(vanta_core) -> StrategiesModuleAdapter:
    """Register the Strategies module with Vanta orchestrator."""
    logger.info("Registering Strategies module with Vanta orchestrator...")
    
    adapter = StrategiesModuleAdapter()
    success = await adapter.initialize(vanta_core)
    
    if success:
        logger.info("Strategies module registration successful")
    else:
        logger.error("Strategies module registration failed")
    
    return adapter


# Export the adapter class for external use
__all__ = ['StrategiesModuleAdapter', 'register_strategies_module']
