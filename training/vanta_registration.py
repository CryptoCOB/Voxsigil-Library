"""
VoxSigil Training Module Registration
====================================

Automatically registers training-related modules with the Vanta orchestrator.
This bridges the existing training implementations with the new modular architecture.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def register_training_modules(vanta_system, config: Optional[Dict[str, Any]] = None):
    """
    Register all training-related modules with Vanta.
    
    Args:
        vanta_system: The VantaSystem instance to register with
        config: Optional configuration dictionary
    """
    config = config or {}
    registration_results = {}
    
    logger.info("Registering training modules with Vanta...")
    
    try:
        from Vanta.integration.module_adapters import module_registry
        
        # Register RAG interfaces
        await register_rag_interfaces(module_registry, config)
        registration_results['rag_interfaces'] = True
        
        # Register other training components
        await register_training_components(module_registry, config)
        registration_results['training_components'] = True
        
        logger.info("Successfully registered all training modules with Vanta")
        return registration_results
        
    except Exception as e:
        logger.error(f"Failed to register training modules: {str(e)}")
        registration_results['error'] = str(e)
        return registration_results


async def register_rag_interfaces(module_registry, config: Dict[str, Any]):
    """Register training RAG interfaces with Vanta."""
    try:
        from training.rag_interface import (
            SupervisorRagInterface,
            SimpleRagInterface,
            VOXSIGIL_RAG_AVAILABLE
        )
        from Vanta.core.fallback_implementations import FallbackRagInterface
        
        # Register SupervisorRagInterface
        if VOXSIGIL_RAG_AVAILABLE:
            supervisor_mapping = {
                'retrieve_documents': 'retrieve_documents',
                'index_document': 'index_document',
                'augment_query': 'augment_query',
                'get_retrieval_stats': 'get_retrieval_stats',
                # Legacy method mappings
                'retrieve_sigils': 'retrieve_sigils',
                'retrieve_context': 'retrieve_context',
                'retrieve_scaffolds': 'retrieve_scaffolds',
                'get_scaffold_definition': 'get_scaffold_definition',
                'get_sigil_by_id': 'get_sigil_by_id'
            }
            
            success = await module_registry.register_class_based_module(
                module_id='training_supervisor_rag',
                module_class=SupervisorRagInterface,
                interface_mapping=supervisor_mapping,
                init_args={
                    'voxsigil_library_path': config.get('voxsigil_library_path'),
                    'embedding_model_name': config.get('embedding_model_name', 'all-MiniLM-L6-v2')
                },
                module_info={
                    'name': 'Training Supervisor RAG Interface',
                    'type': 'training_rag',
                    'description': 'RAG interface for training supervision with VoxSigilRAG',
                    'capabilities': ['document_retrieval', 'sigil_retrieval', 'context_generation']
                }
            )
            
            if success:
                logger.info("Registered SupervisorRagInterface with Vanta")
            else:
                logger.warning("Failed to register SupervisorRagInterface")
        
        # Register SimpleRagInterface (requires a RAG processor)
        simple_mapping = {
            'retrieve_documents': 'retrieve_documents',
            'index_document': 'index_document',
            'augment_query': 'augment_query',
            'get_retrieval_stats': 'get_retrieval_stats',
            # Legacy method mappings
            'retrieve_sigils': 'retrieve_sigils',
            'retrieve_context': 'retrieve_context',
            'retrieve_scaffolds': 'retrieve_scaffolds',
            'get_scaffold_definition': 'get_scaffold_definition',
            'get_sigil_by_id': 'get_sigil_by_id'
        }
          # Note: SimpleRagInterface requires a rag_processor parameter
        # This would typically be provided during specific usage
        logger.info("SimpleRagInterface registration skipped (requires rag_processor)")
        
        # Register FallbackRagInterface for testing
        mock_mapping = {
            'retrieve_documents': 'retrieve_documents',
            'index_document': 'index_document',
            'augment_query': 'augment_query',
            'get_retrieval_stats': 'get_retrieval_stats',
            # Legacy method mappings
            'retrieve_sigils': 'retrieve_sigils',
            'retrieve_context': 'retrieve_context',
            'retrieve_scaffolds': 'retrieve_scaffolds',
            'get_scaffold_definition': 'get_scaffold_definition',
            'get_sigil_by_id': 'get_sigil_by_id'
        }
        
        success = await module_registry.register_class_based_module(
            module_id='training_mock_rag',
            module_class=FallbackRagInterface,
            interface_mapping=mock_mapping,
            init_args={},
            module_info={
                'name': 'Training Mock RAG Interface',
                'type': 'mock_rag',
                'description': 'Mock RAG interface for training development and testing',
                'capabilities': ['mock_retrieval', 'mock_indexing', 'testing']
            }
        )
        
        if success:
            logger.info("Registered FallbackRagInterface with Vanta")
        else:
            logger.warning("Failed to register FallbackRagInterface")
        
    except Exception as e:
        logger.error(f"Error registering RAG interfaces: {str(e)}")
        raise


async def register_training_components(module_registry, config: Dict[str, Any]):
    """Register other training-related components."""
    try:
        # Look for other training components to register
        training_components = [
            'training_supervisor',
            'training_evaluator',
            'training_orchestrator'
        ]
        
        for component in training_components:
            try:
                # Try to dynamically import and register components
                # This would be expanded based on actual training module structure
                logger.debug(f"Checking for training component: {component}")
                
            except ImportError:
                logger.debug(f"Training component {component} not available for registration")
                continue
        
        logger.info("Training component registration check complete")
        
    except Exception as e:
        logger.error(f"Error registering training components: {str(e)}")


async def create_training_adapter(rag_processor=None, interface_type='supervisor'):
    """
    Create a training adapter for immediate use.
    
    Args:
        rag_processor: Optional RAG processor for SimpleRagInterface
        interface_type: Type of interface ('supervisor', 'simple', 'mock')
      Returns:
        Configured adapter ready for Vanta registration
    """
        
    try:
        from training.rag_interface import (
            SupervisorRagInterface,
            SimpleRagInterface,
            VOXSIGIL_RAG_AVAILABLE
        )
        from Vanta.core.fallback_implementations import FallbackRagInterface
        from Vanta.integration.module_adapters import ClassBasedAdapter
        
        if interface_type == 'supervisor' and VOXSIGIL_RAG_AVAILABLE:
            interface_mapping = {
                'retrieve_documents': 'retrieve_documents',
                'index_document': 'index_document',
                'augment_query': 'augment_query',
                'get_retrieval_stats': 'get_retrieval_stats',
                'retrieve_sigils': 'retrieve_sigils',
                'retrieve_context': 'retrieve_context',
                'retrieve_scaffolds': 'retrieve_scaffolds',
                'get_scaffold_definition': 'get_scaffold_definition',
                'get_sigil_by_id': 'get_sigil_by_id'
            }
            
            adapter = ClassBasedAdapter(
                module_id=f'training_rag_{interface_type}',
                module_class=SupervisorRagInterface,
                interface_mapping=interface_mapping,
                init_args={},
                module_info={
                    'name': f'Training {interface_type.title()} RAG Adapter',
                    'type': 'training_rag_adapter',
                    'description': f'Dynamic {interface_type} RAG adapter for training'
                }
            )
            
            return adapter
        
        elif interface_type == 'simple' and rag_processor:
            interface_mapping = {
                'retrieve_documents': 'retrieve_documents',
                'index_document': 'index_document',
                'augment_query': 'augment_query',
                'get_retrieval_stats': 'get_retrieval_stats',
                'retrieve_sigils': 'retrieve_sigils',
                'retrieve_context': 'retrieve_context',
                'retrieve_scaffolds': 'retrieve_scaffolds',
                'get_scaffold_definition': 'get_scaffold_definition',
                'get_sigil_by_id': 'get_sigil_by_id'
            }
            
            adapter = ClassBasedAdapter(
                module_id=f'training_rag_{interface_type}',
                module_class=SimpleRagInterface,
                interface_mapping=interface_mapping,                init_args={'rag_processor': rag_processor},
                module_info={
                    'name': f'Training {interface_type.title()} RAG Adapter',
                    'type': 'training_rag_adapter',
                    'description': f'Dynamic {interface_type} RAG adapter for training'
                }
            )
            
            return adapter
        
        elif interface_type == 'mock':
            interface_mapping = {
                'retrieve_documents': 'retrieve_documents',
                'index_document': 'index_document', 
                'augment_query': 'augment_query',
                'get_retrieval_stats': 'get_retrieval_stats',
                'retrieve_sigils': 'retrieve_sigils',
                'retrieve_context': 'retrieve_context',
                'retrieve_scaffolds': 'retrieve_scaffolds',
                'get_scaffold_definition': 'get_scaffold_definition',
                'get_sigil_by_id': 'get_sigil_by_id'
            }
            
            adapter = ClassBasedAdapter(
                module_id=f'training_rag_{interface_type}',
                module_class=FallbackRagInterface,
                interface_mapping=interface_mapping,
                init_args={},
                module_info={
                    'name': f'Training {interface_type.title()} RAG Adapter',
                    'type': 'training_rag_adapter',
                    'description': f'Dynamic {interface_type} RAG adapter for training'
                }
            )
            
            return adapter
        
        else:
            raise ValueError(f"Unsupported interface type or missing dependencies: {interface_type}")
    
    except Exception as e:
        logger.error(f"Failed to create training adapter: {str(e)}")
        raise


# Convenience function for easy integration
async def integrate_training_with_vanta(vanta_system, config: Optional[Dict[str, Any]] = None):
    """
    High-level function to integrate training modules with Vanta.
    
    Args:
        vanta_system: VantaSystem instance
        config: Optional configuration
    
    Returns:
        Integration results
    """
    try:
        logger.info("Starting training-Vanta integration...")
        
        # Register all training modules
        registration_results = await register_training_modules(vanta_system, config)
        
        # Verify registrations
        system_status = await vanta_system.get_detailed_status()
        registered_modules = system_status.get('registered_modules', [])
        
        training_modules = [m for m in registered_modules if 'training' in m]
        
        integration_results = {
            'registration_results': registration_results,
            'registered_training_modules': training_modules,
            'total_training_modules': len(training_modules),
            'integration_successful': len(training_modules) > 0
        }
        
        logger.info(f"Training-Vanta integration complete. Registered {len(training_modules)} training modules.")
        
        return integration_results
        
    except Exception as e:
        logger.error(f"Training-Vanta integration failed: {str(e)}")
        return {
            'integration_successful': False,
            'error': str(e)
        }
