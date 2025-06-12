"""
BLT Module Registration with Vanta
==================================

Registers BLT-enhanced modules with the Vanta orchestrator system.
Provides integration for BLT-enhanced RAG and other BLT components.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import importlib
import inspect
from Vanta.integration.module_adapters import LegacyModuleAdapter

logger = logging.getLogger(__name__)

async def register_blt_modules(vanta_system, config: Optional[Dict[str, Any]] = None):
    """
    Register all BLT-enhanced modules with Vanta.
    
    Args:
        vanta_system: The VantaSystem instance to register with
        config: Optional configuration dictionary
    """
    config = config or {}
    registration_results = {}
    
    logger.info("Registering BLT modules with Vanta...")
    
    try:
        from Vanta.integration.module_adapters import module_registry
        
        # Register BLT RAG interface
        await register_blt_rag_interface(module_registry, config)
        registration_results['blt_rag_interface'] = True

        # Register BLT middleware components
        await register_blt_middleware(module_registry, config)
        registration_results['blt_middleware'] = True

        # Register TinyLlama integration
        await register_tinyllama_integration(module_registry, config)
        registration_results['tinyllama_integration'] = True

        # Register any remaining BLT modules in this directory
        await register_all_blt_files(module_registry)
        registration_results['blt_files'] = True
        
        logger.info("Successfully registered all BLT modules with Vanta")
        return registration_results
        
    except Exception as e:
        logger.error(f"Failed to register BLT modules: {str(e)}")
        registration_results['error'] = str(e)
        return registration_results


async def register_blt_rag_interface(module_registry, config: Dict[str, Any]):
    """Register BLT-enhanced RAG interface with Vanta."""
    try:
        from BLT.blt_supervisor_integration import (
            BLTSupervisorRagInterface,
            COMPONENTS_AVAILABLE
        )
        
        if not COMPONENTS_AVAILABLE:
            logger.warning("BLT components not available, skipping BLT RAG registration")
            return
        
        # Define interface mapping for BLT RAG
        blt_rag_mapping = {
            # Unified interface methods
            'retrieve_documents': 'retrieve_documents',
            'index_document': 'index_document', 
            'augment_query': 'augment_query',
            'get_retrieval_stats': 'get_retrieval_stats',
            # Legacy/BLT-specific methods
            'retrieve_sigils': 'retrieve_sigils',
            'retrieve_context': 'retrieve_context',
            'initialize_components': 'initialize_components',
            'create_supervisor': 'create_supervisor'
        }
        
        # BLT configuration
        blt_config = {
            'entropy_threshold': config.get('blt_entropy_threshold', 0.25),
            'blt_hybrid_weight': config.get('blt_hybrid_weight', 0.7),
            'entropy_router_fallback': config.get('entropy_router_fallback', 'token_based'),
            'cache_ttl_seconds': config.get('cache_ttl_seconds', 300),
            'log_level': config.get('log_level', 'INFO')
        }
        
        init_args = {
            'voxsigil_library_path': config.get('voxsigil_library_path'),
            'embedding_model_name': config.get('embedding_model_name', 'all-MiniLM-L6-v2'),
            'blt_config': blt_config
        }
        
        success = await module_registry.register_class_based_module(
            module_id='blt_enhanced_rag',
            module_class=BLTSupervisorRagInterface,
            interface_mapping=blt_rag_mapping,
            init_args=init_args,
            module_info={
                'name': 'BLT Enhanced RAG Interface',
                'type': 'blt_rag',
                'description': 'BLT-enhanced RAG with entropy routing and byte-level transformers',
                'capabilities': [
                    'enhanced_retrieval',
                    'entropy_routing', 
                    'blt_encoding',
                    'patch_validation',
                    'hybrid_embedding',
                    'context_optimization'
                ],
                'blt_features': {
                    'entropy_threshold': blt_config['entropy_threshold'],
                    'hybrid_weight': blt_config['blt_hybrid_weight'],
                    'cache_enabled': True,
                    'patch_compression': True
                }
            }
        )
        
        if success:
            logger.info("Successfully registered BLTSupervisorRagInterface with Vanta")
        else:
            logger.warning("Failed to register BLTSupervisorRagInterface")
            
    except Exception as e:
        logger.error(f"Error registering BLT RAG interface: {str(e)}")
        raise


async def register_blt_middleware(module_registry, config: Dict[str, Any]):
    """Register BLT middleware components with Vanta."""
    try:
        from BLT.blt_supervisor_integration import COMPONENTS_AVAILABLE
        
        if not COMPONENTS_AVAILABLE:
            logger.warning("BLT components not available, skipping middleware registration")
            return
        
        # Try to import BLT middleware components
        try:
            from BLT.hybrid_blt import (
                HybridMiddlewareConfig,
                EntropyRouter,
                ByteLatentTransformerEncoder
            )
            
            # Register HybridMiddlewareConfig
            middleware_mapping = {
                'get_config': '__dict__',  # Expose configuration
                'update_config': '__setattr__'
            }
            
            config_init_args = {
                'entropy_threshold': config.get('blt_entropy_threshold', 0.25),
                'blt_hybrid_weight': config.get('blt_hybrid_weight', 0.7),
                'entropy_router_fallback': config.get('entropy_router_fallback', 'token_based'),
                'cache_ttl_seconds': config.get('cache_ttl_seconds', 300),
                'log_level': config.get('log_level', 'INFO')
            }
            
            success = await module_registry.register_class_based_module(
                module_id='blt_middleware_config',
                module_class=HybridMiddlewareConfig,
                interface_mapping=middleware_mapping,
                init_args=config_init_args,
                module_info={
                    'name': 'BLT Hybrid Middleware Configuration',
                    'type': 'blt_middleware',
                    'description': 'Configuration management for BLT hybrid middleware',
                    'capabilities': ['config_management', 'entropy_thresholds', 'hybrid_weights']
                }
            )
            
            if success:
                logger.info("Registered BLT Middleware Configuration with Vanta")
            
            # Register EntropyRouter
            router_mapping = {
                'route': 'route',
                'calculate_entropy': 'calculate_entropy' if hasattr(EntropyRouter, 'calculate_entropy') else None
            }
            router_mapping = {k: v for k, v in router_mapping.items() if v is not None}
            
            success = await module_registry.register_class_based_module(
                module_id='blt_entropy_router',
                module_class=EntropyRouter,
                interface_mapping=router_mapping,
                init_args={},
                module_info={
                    'name': 'BLT Entropy Router',
                    'type': 'blt_router',
                    'description': 'Entropy-based routing for BLT processing',
                    'capabilities': ['entropy_calculation', 'adaptive_routing', 'fallback_routing']
                }
            )
            
            if success:
                logger.info("Registered BLT Entropy Router with Vanta")
            
            # Register ByteLatentTransformerEncoder  
            encoder_mapping = {
                'create_patches': 'create_patches',
                'encode': 'encode' if hasattr(ByteLatentTransformerEncoder, 'encode') else None,
                'calculate_similarity': 'calculate_similarity' if hasattr(ByteLatentTransformerEncoder, 'calculate_similarity') else None
            }
            encoder_mapping = {k: v for k, v in encoder_mapping.items() if v is not None}
            
            success = await module_registry.register_class_based_module(
                module_id='blt_encoder',
                module_class=ByteLatentTransformerEncoder,
                interface_mapping=encoder_mapping,
                init_args={},
                module_info={
                    'name': 'BLT Byte Latent Transformer Encoder',
                    'type': 'blt_encoder',
                    'description': 'Byte-level latent transformer encoding for enhanced embeddings',
                    'capabilities': ['patch_creation', 'byte_encoding', 'similarity_calculation']
                }
            )
            
            if success:
                logger.info("Registered BLT Encoder with Vanta")
                
        except ImportError as e:
            logger.warning(f"Some BLT middleware components not available: {e}")
            
    except Exception as e:
        logger.error(f"Error registering BLT middleware: {str(e)}")


async def register_tinyllama_integration(module_registry, config: Dict[str, Any]):
    """Register TinyLlama integration components with Vanta."""
    try:
        from BLT.blt_supervisor_integration import (
            TinyLlamaIntegration,
            COMPONENTS_AVAILABLE
        )
        
        if not COMPONENTS_AVAILABLE:
            logger.warning("BLT components not available, skipping TinyLlama registration")
            return
        
        # TinyLlama integration mapping
        tinyllama_mapping = {
            'create_supervisor': 'create_supervisor',
            'validate_integration': 'validate_integration'
        }
        
        # TinyLlama configuration
        tinyllama_config = {
            'model_name': config.get('tinyllama_model', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'),
            'blt_config': {
                'entropy_threshold': config.get('blt_entropy_threshold', 0.25),
                'blt_hybrid_weight': config.get('blt_hybrid_weight', 0.7),
                'entropy_router_fallback': config.get('entropy_router_fallback', 'token_based'),
                'cache_ttl_seconds': config.get('cache_ttl_seconds', 300),
                'log_level': config.get('log_level', 'INFO')
            }
        }
        
        init_args = {
            'voxsigil_library_path': config.get('voxsigil_library_path'),
            'model_name': tinyllama_config['model_name'],
            'blt_config': tinyllama_config['blt_config']
        }
        
        success = await module_registry.register_class_based_module(
            module_id='blt_tinyllama_integration',
            module_class=TinyLlamaIntegration,
            interface_mapping=tinyllama_mapping,
            init_args=init_args,
            module_info={
                'name': 'BLT-TinyLlama Integration',
                'type': 'blt_llm_integration',
                'description': 'Integration of TinyLlama with BLT-enhanced RAG',
                'capabilities': [
                    'llm_integration',
                    'supervisor_creation',
                    'integration_validation',
                    'blt_enhanced_generation'
                ],
                'model_info': {
                    'model_name': tinyllama_config['model_name'],
                    'blt_enhanced': True,
                    'entropy_routing': True
                }
            }
        )
        
        if success:
            logger.info("Successfully registered TinyLlama Integration with Vanta")
        else:
            logger.warning("Failed to register TinyLlama Integration")
            
    except Exception as e:
        logger.error(f"Error registering TinyLlama integration: {str(e)}")


async def register_all_blt_files(module_registry) -> None:
    """Auto-register all BLT modules in this directory."""
    blt_dir = Path(__file__).parent

    for py_file in blt_dir.glob("*.py"):
        stem = py_file.stem
        if stem in {"__init__", "vanta_registration"}:
            continue

        module_name = f"BLT.{stem}"
        try:
            module = importlib.import_module(module_name)
            functions = {
                name: name
                for name, obj in inspect.getmembers(module, inspect.isfunction)
                if not name.startswith("_")
            }

            if not functions:
                # Skip modules without callable functions
                continue

            adapter = LegacyModuleAdapter(
                module_id=f"blt_{stem}",
                legacy_module=module,
                method_mapping=functions,
                module_info={
                    "name": stem,
                    "type": "blt_module",
                    "description": f"BLT module {stem}",
                },
            )
            await module_registry.register_custom_adapter(f"blt_{stem}", adapter)
            logger.info(f"Registered BLT module file: {stem}")
        except Exception as exc:
            logger.warning(f"Failed to register BLT module {stem}: {exc}")


async def create_blt_adapter(adapter_type='rag', config: Optional[Dict[str, Any]] = None):
    """
    Create a BLT adapter for immediate use.
    
    Args:
        adapter_type: Type of adapter ('rag', 'middleware', 'tinyllama')
        config: Optional configuration
    
    Returns:
        Configured adapter ready for Vanta registration
    """
    config = config or {}
    
    try:
        from BLT.blt_supervisor_integration import COMPONENTS_AVAILABLE
        from Vanta.integration.module_adapters import ClassBasedAdapter
        
        if not COMPONENTS_AVAILABLE:
            raise ValueError("BLT components not available")
        
        if adapter_type == 'rag':
            from BLT.blt_supervisor_integration import BLTSupervisorRagInterface
            
            interface_mapping = {
                'retrieve_documents': 'retrieve_documents',
                'index_document': 'index_document',
                'augment_query': 'augment_query',
                'get_retrieval_stats': 'get_retrieval_stats',
                'retrieve_sigils': 'retrieve_sigils',
                'retrieve_context': 'retrieve_context',
                'initialize_components': 'initialize_components'
            }
            
            init_args = {
                'voxsigil_library_path': config.get('voxsigil_library_path'),
                'embedding_model_name': config.get('embedding_model_name', 'all-MiniLM-L6-v2'),
                'blt_config': config.get('blt_config', {})
            }
            
            adapter = ClassBasedAdapter(
                module_id=f'blt_rag_adapter_{id(config)}',
                module_class=BLTSupervisorRagInterface,
                interface_mapping=interface_mapping,
                init_args=init_args,
                module_info={
                    'name': 'Dynamic BLT RAG Adapter',
                    'type': 'blt_rag_adapter',
                    'description': 'Dynamic BLT-enhanced RAG adapter'
                }
            )
            
            return adapter
        
        elif adapter_type == 'tinyllama':
            from BLT.blt_supervisor_integration import TinyLlamaIntegration
            
            interface_mapping = {
                'create_supervisor': 'create_supervisor',
                'validate_integration': 'validate_integration'
            }
            
            init_args = {
                'voxsigil_library_path': config.get('voxsigil_library_path'),
                'model_name': config.get('model_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'),
                'blt_config': config.get('blt_config', {})
            }
            
            adapter = ClassBasedAdapter(
                module_id=f'blt_tinyllama_adapter_{id(config)}',
                module_class=TinyLlamaIntegration,
                interface_mapping=interface_mapping,
                init_args=init_args,
                module_info={
                    'name': 'Dynamic BLT-TinyLlama Adapter',
                    'type': 'blt_tinyllama_adapter', 
                    'description': 'Dynamic BLT-TinyLlama integration adapter'
                }
            )
            
            return adapter
        
        else:
            raise ValueError(f"Unsupported BLT adapter type: {adapter_type}")
    
    except Exception as e:
        logger.error(f"Failed to create BLT adapter: {str(e)}")
        raise


# Convenience function for easy integration  
async def integrate_blt_with_vanta(vanta_system, config: Optional[Dict[str, Any]] = None):
    """
    High-level function to integrate BLT modules with Vanta.
    
    Args:
        vanta_system: VantaSystem instance
        config: Optional configuration
    
    Returns:
        Integration results
    """
    try:
        logger.info("Starting BLT-Vanta integration...")
        
        # Register all BLT modules
        registration_results = await register_blt_modules(vanta_system, config)
        
        # Verify registrations
        system_status = await vanta_system.get_detailed_status()
        registered_modules = system_status.get('registered_modules', [])
        
        blt_modules = [m for m in registered_modules if 'blt' in m.lower()]
        
        integration_results = {
            'registration_results': registration_results,
            'registered_blt_modules': blt_modules,
            'total_blt_modules': len(blt_modules),
            'integration_successful': len(blt_modules) > 0
        }
        
        logger.info(f"BLT-Vanta integration complete. Registered {len(blt_modules)} BLT modules.")
        
        return integration_results
        
    except Exception as e:
        logger.error(f"BLT-Vanta integration failed: {str(e)}")
        return {
            'integration_successful': False,
            'error': str(e)
        }
