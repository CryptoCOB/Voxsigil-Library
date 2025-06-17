#!/usr/bin/env python
"""
Dependency Health Monitor

Monitors pip packages, outdated libraries, and system dependencies.
"""

import asyncio
import json
import logging
import time

from core.base import BaseCore
from core.vanta_registration import CognitiveMeshRole, vanta_core_module

logger = logging.getLogger("VoxSigil.Dependency.Health")


@vanta_core_module(
    name="dependency_health_monitor",
    subsystem="system",
    mesh_role=CognitiveMeshRole.MONITOR,
    description="Dependency and package health monitoring",
    capabilities=["package_tracking", "outdated_detection", "vulnerability_check"],
)
class DependencyHealthMonitor(BaseCore):
    """
    Monitors system dependencies and package health.
    """

    def __init__(self):
        super().__init__()
        self.check_interval = 300  # 5 minutes
        self.packages = {}
        self.outdated_packages = {}
        self.system_info = {}

    async def initialize_subsystem(self, core):
        """Initialize the dependency monitor."""
        self.core = core
        logger.info("Dependency health monitor initialized")

        # Initial check
        await self._check_dependencies()

        # Start periodic checks
        asyncio.create_task(self._periodic_check())

    async def _periodic_check(self):
        """Run dependency checks periodically."""
        while True:
            try:
                await self._check_dependencies()
                await self._publish_status()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in periodic dependency check: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute on error

    async def _check_dependencies(self):
        """Check all dependencies."""
        logger.info("Checking dependencies...")

        # Check pip packages
        await self._check_pip_packages()

        # Check for outdated packages
        await self._check_outdated_packages()

        # Check system info
        await self._check_system_info()

    async def _check_pip_packages(self):
        """Check installed pip packages."""
        try:
            # Get installed packages
            process = await asyncio.create_subprocess_exec(
                "pip",
                "list",
                "--format=json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                packages_data = json.loads(stdout.decode())
                self.packages = {pkg["name"]: pkg["version"] for pkg in packages_data}
                logger.info(f"Found {len(self.packages)} installed packages")
            else:
                logger.error(f"Failed to get pip packages: {stderr.decode()}")

        except Exception as e:
            logger.error(f"Error checking pip packages: {e}")

    async def _check_outdated_packages(self):
        """Check for outdated packages."""
        try:
            # Get outdated packages
            process = await asyncio.create_subprocess_exec(
                "pip",
                "list",
                "--outdated",
                "--format=json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                outdated_data = json.loads(stdout.decode())
                self.outdated_packages = {
                    pkg["name"]: {
                        "current": pkg["version"],
                        "latest": pkg["latest_version"],
                        "type": pkg.get("latest_filetype", "wheel"),
                    }
                    for pkg in outdated_data
                }
                logger.info(f"Found {len(self.outdated_packages)} outdated packages")
            else:
                logger.warning(f"Failed to get outdated packages: {stderr.decode()}")

        except Exception as e:
            logger.error(f"Error checking outdated packages: {e}")

    async def _check_system_info(self):
        """Check system dependency info."""
        try:
            self.system_info = {
                "python_version": await self._get_python_version(),
                "pip_version": await self._get_pip_version(),
                "cuda_available": await self._check_cuda(),
                "gpu_driver": await self._get_gpu_driver(),
            }

        except Exception as e:
            logger.error(f"Error checking system info: {e}")

    async def _get_python_version(self) -> str:
        """Get Python version."""
        try:
            process = await asyncio.create_subprocess_exec(
                "python",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            return stdout.decode().strip()
        except Exception:
            return "Unknown"

    async def _get_pip_version(self) -> str:
        """Get pip version."""
        try:
            process = await asyncio.create_subprocess_exec(
                "pip", "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            return stdout.decode().strip()
        except Exception:
            return "Unknown"

    async def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    async def _get_gpu_driver(self) -> str:
        """Get GPU driver version."""
        try:
            # Try nvidia-smi first
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return f"NVIDIA {stdout.decode().strip()}"
            else:
                return "No NVIDIA GPU detected"

        except FileNotFoundError:
            return "nvidia-smi not found"
        except Exception as e:
            return f"Error: {e}"

    async def _publish_status(self):
        """Publish dependency status."""
        if hasattr(self, "core") and self.core:
            status_data = {
                "timestamp": time.time(),
                "total_packages": len(self.packages),
                "outdated_count": len(self.outdated_packages),
                "packages": self.packages,
                "outdated": self.outdated_packages,
                "system": self.system_info,
            }

            self.core.bus.publish("dependency.health", status_data)


# UI Specification for bridge integration
def get_dependency_ui_spec():
    """Return UI specification for bridge integration."""
    return {
        "id": "dependency_health",
        "ui_spec": {
            "tab": "Dependency Health",
            "widget": "DependencyPanel",
            "stream": True,
            "stream_topic": "dependency.health",
            "icon": "📦",
        },
    }


if __name__ == "__main__":
    # Test the dependency monitor
    import asyncio

    async def test_dependency_monitor():
        monitor = DependencyHealthMonitor()
        await monitor._check_dependencies()
        print(f"Found {len(monitor.packages)} packages")
        print(f"Outdated: {len(monitor.outdated_packages)}")
        print(f"System: {monitor.system_info}")

    asyncio.run(test_dependency_monitor())
