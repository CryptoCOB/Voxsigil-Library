"""
ARC Module Registration with Vanta
=================================

Registers ARC-related modules (Abstraction and Reasoning Corpus helpers)
with the Vanta orchestrator. This provides dynamic loading of the ARC
reasoner, dataset utilities and GridFormer components.
"""

import logging
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, Type, Union

logger = logging.getLogger("Vanta.ARCRegistration")


class ARCModuleAdapter:
    """Adapter for registering ARC modules as Vanta components."""

    def __init__(self, module_id: str, arc_object: Union[Type, object], description: str):
        self.module_id = module_id
        self.arc_object = arc_object
        self.description = description
        self.instance = None

    async def initialize(self, vanta_core) -> bool:
        """Initialize the underlying ARC object if it is a class."""
        try:
            if isinstance(self.arc_object, type):
                # Instantiate with optional vanta_core if supported
                try:
                    self.instance = self.arc_object(vanta_core=vanta_core)
                except Exception:
                    self.instance = self.arc_object()
            else:
                self.instance = self.arc_object
            logger.info(f"ARC module {self.module_id} initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ARC module {self.module_id}: {e}")
            return False

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not self.instance:
            return {"error": f"ARC module {self.module_id} not initialized"}
        handler = None
        if hasattr(self.instance, 'handle_request'):
            handler = self.instance.handle_request
        elif hasattr(self.instance, 'process'):
            handler = self.instance.process
        if handler:
            try:
                result = await handler(request)
            except TypeError:
                result = handler(request)
            return {"module": self.module_id, "result": result}
        return {"module": self.module_id, "message": "request processed"}

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "type": "arc_module",
            "description": self.description,
            "module": getattr(self.arc_object, '__name__', str(self.arc_object))
        }


def import_arc_object(name: str) -> Optional[Union[Type, object]]:
    """Import ARC class or module based on name."""
    try:
        module = importlib.import_module(f'ARC.{name}')
        class_mapping = {
            'arc_reasoner': 'ARCReasoner',
            'arc_data_processor': 'ARCGridDataProcessor',
            'arc_gridformer_blt_adapter': 'GridFormerBLTAdapter',
            'arc_gridformer_core': 'GRID_Former',
            'arc_integration': 'HybridARCSolver',
            'arc_grid_former_pipeline': 'GridFormerQuantizer',
        }
        class_name = class_mapping.get(name)
        if class_name and hasattr(module, class_name):
            return getattr(module, class_name)
        # fallback: first exported class
        for attr in dir(module):
            if attr[0].isupper() and hasattr(getattr(module, attr), '__init__'):
                return getattr(module, attr)
        return module
    except Exception as e:
        logger.error(f"Failed to import ARC module {name}: {e}")
        return None


async def register_arc_modules() -> tuple[int, int]:
    """Auto-register all ARC modules with Vanta."""
    from Vanta import get_vanta_core_instance

    vanta = get_vanta_core_instance()

    arc_dir = Path(__file__).parent
    arc_modules = [
        p.stem for p in arc_dir.glob("*.py")
        if p.stem not in {"__init__", "vanta_registration"}
    ]

    registered = 0
    failed = 0

    logger.info(
        f"Starting ARC module registration ({len(arc_modules)} modules)..."
    )

    for name in arc_modules:
        arc_obj = import_arc_object(name)
        if not arc_obj:
            logger.warning(f"Skipping ARC component {name}")
            failed += 1
            continue
        adapter = ARCModuleAdapter(
            module_id=f"arc_{name}",
            arc_object=arc_obj,
            description=f"ARC component: {name}",
        )
        try:
            await vanta.register_module(f"arc_{name}", adapter)
            registered += 1
            logger.info(f"Registered ARC module: {name}")
        except Exception as e:
            logger.error(f"Failed to register ARC module {name}: {e}")
            failed += 1

    logger.info(f"ARC registration complete: {registered} successful, {failed} failed")
    return registered, failed


__all__ = ['register_arc_modules', 'ARCModuleAdapter']
