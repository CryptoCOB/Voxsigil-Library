"""
Tags Module Registration - HOLO-1.5 Enhanced Cognitive Mesh

Every module in this package is imported and registered with UnifiedVantaCore
via the decorator below. Edit ONLY the metadata fields—keep the class + async
signature identical so the master orchestrator can introspect it.
"""

from core.base import BaseCore, CognitiveMeshRole, vanta_core_module
import logging

logger = logging.getLogger(__name__)

@vanta_core_module(
    name="tag_registry",
    subsystem="symbols",
    mesh_role=CognitiveMeshRole.BINDER,
    description=(
        "Tag registry for namespace management, taxonomy organization, "
        "semantic embedding, and symbolic relationship mapping"
    ),
    capabilities=[
        "namespace",
        "taxonomy",
        "embedding",
        "tag_classification",
        "semantic_binding",
        "relationship_mapping",
        "symbolic_tagging",
        "category_management"
    ],
    cognitive_load=2.0,
    symbolic_depth=4,
    collaboration_patterns=[
        "symbolic_binding",
        "semantic_organization",
        "taxonomy_management",
        "namespace_coordination"
    ],
)
class TagsModule(BaseCore):
    """
    Tag registry registration wrapper. Real tag implementations live 
    in tags/*.py but vanta_core only needs a singleton object to expose them.
    """

    async def initialize_subsystem(self, vanta_core):
        """
        Called once on startup. Do lightweight init only.
        Heavy I/O or GPU ops should be lazy-loaded the first
        time execute_* is invoked.
        """
        await super().initialize_subsystem(vanta_core)
        logger.info("TagsModule initialised.")

    async def execute_task(self, payload: dict) -> dict:
        """
        Generic dispatch for tag sub-tasks.
        payload = { "op": "classify", "content": "...", "namespace": "cognitive" }
        """
        op = payload.get("op")
        if op == "classify":
            # Example tag classification
            content = payload.get("content", "")
            namespace = payload.get("namespace", "default")
            return {"tags": [f"{namespace}_tag", "auto_classified"], "confidence": 0.85}
        elif op == "embed":
            # Example semantic embedding
            tags = payload.get("tags", [])
            return {"embeddings": [f"embed_{tag}" for tag in tags], "dimension": 384}
        raise ValueError(f"Unsupported tag op: {op}")

# Registration function for master orchestrator
async def register(vanta_core):
    """Register the tags module with Vanta orchestrator."""
    module = TagsModule(vanta_core, {})
    await module.initialize_subsystem(vanta_core)
    vanta_core.registry.register(module)
    return module

__all__ = ["register", "TagsModule"]
