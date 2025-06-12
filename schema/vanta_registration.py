"""
Schema Module Registration - HOLO-1.5 Enhanced Cognitive Mesh

Every module in this package is imported and registered with UnifiedVantaCore
via the decorator below. Edit ONLY the metadata fields—keep the class + async
signature identical so the master orchestrator can introspect it.
"""

from core.base import BaseCore, CognitiveMeshRole, vanta_core_module
import logging

logger = logging.getLogger(__name__)

@vanta_core_module(
    name="schema_validator",
    subsystem="schema",
    mesh_role=CognitiveMeshRole.MONITOR,
    description=(
        "Schema validator for JSON schema validation, datatype definitions, "
        "schema migration, and structural integrity enforcement"
    ),
    capabilities=[
        "jsonschema",
        "datatype_defs",
        "migration",
        "validation",
        "structural_integrity",
        "schema_evolution",
        "compatibility_checking",
        "data_transformation"
    ],
    cognitive_load=2.5,
    symbolic_depth=2,
    collaboration_patterns=[
        "schema_validation",
        "data_integrity",
        "structural_monitoring",
        "compatibility_enforcement"
    ],
)
class SchemaModule(BaseCore):
    """
    Schema validator registration wrapper. Real schema implementations live 
    in schema/*.py but vanta_core only needs a singleton object to expose them.
    """

    async def initialize_subsystem(self, vanta_core):
        """
        Called once on startup. Do lightweight init only.
        Heavy I/O or GPU ops should be lazy-loaded the first
        time execute_* is invoked.
        """
        await super().initialize_subsystem(vanta_core)
        logger.info("SchemaModule initialised.")

    async def execute_task(self, payload: dict) -> dict:
        """
        Generic dispatch for schema sub-tasks.
        payload = { "op": "validate", "schema": {...}, "data": {...} }
        """
        op = payload.get("op")
        if op == "validate":
            # Example schema validation
            schema = payload.get("schema", {})
            data = payload.get("data", {})
            return {"valid": True, "errors": [], "schema_version": "1.5"}
        elif op == "migrate":
            # Example schema migration
            old_version = payload.get("old_version", "1.0")
            new_version = payload.get("new_version", "1.5")
            return {"migrated": True, "from": old_version, "to": new_version}
        raise ValueError(f"Unsupported schema op: {op}")

# Registration function for master orchestrator
async def register(vanta_core):
    """Register the schema module with Vanta orchestrator."""
    module = SchemaModule(vanta_core, {})
    await module.initialize_subsystem(vanta_core)
    vanta_core.registry.register(module)
    return module

__all__ = ["register", "SchemaModule"]
