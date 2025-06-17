#!/usr/bin/env python
"""
Dataset Manager Agent

Monitors dataset status, versions, licenses, and triggers re-indexing operations.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from core.base import BaseCore
from core.vanta_registration import CognitiveMeshRole, vanta_core_module

logger = logging.getLogger("VoxSigil.Dataset.Manager")


@vanta_core_module(
    name="dataset_manager",
    subsystem="data",
    mesh_role=CognitiveMeshRole.MONITOR,
    description="Dataset monitoring and management agent",
    capabilities=["dataset_tracking", "version_control", "license_compliance", "re_indexing"],
)
class DatasetManager(BaseCore):
    """
    Dataset management agent that tracks dataset status, versions, and handles re-indexing.
    """

    def __init__(self):
        super().__init__()
        self.datasets = {}
        self.status_interval = 10  # 10 seconds
        self.dataset_dirs = ["data", "datasets", "training_data", "ARC/data", "tests/data"]

    async def initialize_subsystem(self, core):
        """Initialize the dataset manager."""
        self.core = core
        logger.info("Dataset manager initialized")

        # Initial scan
        await self._scan_datasets()

        # Start periodic status updates
        asyncio.create_task(self._periodic_status_update())

    async def _periodic_status_update(self):
        """Publish dataset status periodically."""
        while True:
            try:
                await self._scan_datasets()
                await self._publish_status()
                await asyncio.sleep(self.status_interval)
            except Exception as e:
                logger.error(f"Error in periodic dataset status update: {e}")
                await asyncio.sleep(30)  # Retry in 30 seconds on error

    async def _scan_datasets(self):
        """Scan for datasets and update status."""
        self.datasets = {}

        for dataset_dir in self.dataset_dirs:
            dataset_path = Path(dataset_dir)
            if dataset_path.exists() and dataset_path.is_dir():
                await self._scan_directory(dataset_path)

    async def _scan_directory(self, directory: Path):
        """Scan a directory for datasets."""
        try:
            for item in directory.iterdir():
                if item.is_dir():
                    dataset_info = await self._analyze_dataset(item)
                    if dataset_info:
                        self.datasets[str(item)] = dataset_info
                elif item.suffix.lower() in [".json", ".jsonl", ".csv", ".tsv", ".parquet"]:
                    dataset_info = await self._analyze_dataset_file(item)
                    if dataset_info:
                        self.datasets[str(item)] = dataset_info

        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")

    async def _analyze_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Analyze a dataset directory."""
        try:
            dataset_info = {
                "path": str(dataset_path),
                "type": "directory",
                "size_mb": 0,
                "file_count": 0,
                "last_modified": 0,
                "license": "unknown",
                "version": "unknown",
                "description": "",
                "status": "ok",
            }

            # Count files and calculate size
            total_size = 0
            file_count = 0
            last_modified = 0

            for file_path in dataset_path.rglob("*"):
                if file_path.is_file():
                    file_count += 1
                    stat = file_path.stat()
                    total_size += stat.st_size
                    last_modified = max(last_modified, stat.st_mtime)

            dataset_info["size_mb"] = round(total_size / (1024 * 1024), 2)
            dataset_info["file_count"] = file_count
            dataset_info["last_modified"] = last_modified

            # Look for metadata files
            metadata_files = ["README.md", "LICENSE", "dataset_info.json", "metadata.json"]
            for metadata_file in metadata_files:
                metadata_path = dataset_path / metadata_file
                if metadata_path.exists():
                    await self._parse_metadata(metadata_path, dataset_info)

            return dataset_info

        except Exception as e:
            logger.error(f"Error analyzing dataset {dataset_path}: {e}")
            return None

    async def _analyze_dataset_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single dataset file."""
        try:
            stat = file_path.stat()

            dataset_info = {
                "path": str(file_path),
                "type": "file",
                "format": file_path.suffix.lower(),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "last_modified": stat.st_mtime,
                "license": "unknown",
                "version": "unknown",
                "rows": 0,
                "status": "ok",
            }

            # Try to count rows for text files
            if file_path.suffix.lower() in [".json", ".jsonl", ".csv", ".tsv"]:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        if file_path.suffix.lower() in [".jsonl"]:
                            dataset_info["rows"] = sum(1 for _ in f)
                        elif file_path.suffix.lower() in [".csv", ".tsv"]:
                            dataset_info["rows"] = sum(1 for _ in f) - 1  # Subtract header
                        elif file_path.suffix.lower() == ".json":
                            data = json.load(f)
                            if isinstance(data, list):
                                dataset_info["rows"] = len(data)
                            elif isinstance(data, dict) and "data" in data:
                                dataset_info["rows"] = len(data["data"])
                except Exception:
                    pass  # Ignore parsing errors

            return dataset_info

        except Exception as e:
            logger.error(f"Error analyzing dataset file {file_path}: {e}")
            return None

    async def _parse_metadata(self, metadata_path: Path, dataset_info: Dict[str, Any]):
        """Parse metadata from a file."""
        try:
            if metadata_path.name == "LICENSE":
                with open(metadata_path, "r", encoding="utf-8") as f:
                    license_text = f.read().strip()
                    if "MIT" in license_text:
                        dataset_info["license"] = "MIT"
                    elif "Apache" in license_text:
                        dataset_info["license"] = "Apache-2.0"
                    elif "GPL" in license_text:
                        dataset_info["license"] = "GPL"
                    else:
                        dataset_info["license"] = "custom"

            elif metadata_path.suffix == ".json":
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    dataset_info.update(
                        {
                            "version": metadata.get("version", "unknown"),
                            "description": metadata.get("description", ""),
                            "license": metadata.get("license", dataset_info["license"]),
                        }
                    )

            elif metadata_path.name == "README.md":
                with open(metadata_path, "r", encoding="utf-8") as f:
                    readme_content = f.read()
                    # Extract description from first paragraph
                    lines = readme_content.split("\n")
                    for line in lines:
                        if line.strip() and not line.startswith("#"):
                            dataset_info["description"] = line.strip()[:200]
                            break

        except Exception as e:
            logger.error(f"Error parsing metadata from {metadata_path}: {e}")

    async def _publish_status(self):
        """Publish current dataset status."""
        if hasattr(self, "core") and self.core:
            status_data = {
                "timestamp": time.time(),
                "total_datasets": len(self.datasets),
                "total_size_mb": sum(ds.get("size_mb", 0) for ds in self.datasets.values()),
                "datasets": self.datasets,
            }

            self.core.bus.publish("dataset.status", status_data)

    async def trigger_reindex(self, dataset_path: str):
        """Trigger re-indexing of a specific dataset."""
        logger.info(f"Triggering re-index for dataset: {dataset_path}")

        # Simulate re-indexing process
        if hasattr(self, "core") and self.core:
            self.core.bus.publish(
                "dataset.reindex.started", {"dataset_path": dataset_path, "timestamp": time.time()}
            )

            # Re-scan the specific dataset
            dataset_path_obj = Path(dataset_path)
            if dataset_path_obj.exists():
                if dataset_path_obj.is_dir():
                    dataset_info = await self._analyze_dataset(dataset_path_obj)
                else:
                    dataset_info = await self._analyze_dataset_file(dataset_path_obj)

                if dataset_info:
                    self.datasets[dataset_path] = dataset_info

            self.core.bus.publish(
                "dataset.reindex.completed",
                {"dataset_path": dataset_path, "timestamp": time.time(), "success": True},
            )


# UI Specification for bridge integration
def get_dataset_ui_spec():
    """Return UI specification for bridge integration."""
    return {
        "id": "dataset_manager",
        "ui_spec": {
            "tab": "Dataset Manager",
            "widget": "DatasetPanel",
            "stream": True,
            "stream_topic": "dataset.status",
            "icon": "📊",
        },
    }


if __name__ == "__main__":
    # Test the dataset manager
    import asyncio

    async def test_dataset_manager():
        manager = DatasetManager()
        await manager._scan_datasets()
        print(f"Found {len(manager.datasets)} datasets")
        for path, info in manager.datasets.items():
            print(
                f"  {path}: {info['size_mb']} MB, {info.get('file_count', info.get('rows', 0))} items"
            )

    asyncio.run(test_dataset_manager())
