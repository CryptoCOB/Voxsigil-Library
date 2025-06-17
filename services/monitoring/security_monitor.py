#!/usr/bin/env python
"""
Security & Compliance Monitor Agent

Monitors security status, CVE scans, permission errors, and compliance issues.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from core.base import BaseCore
from core.vanta_registration import CognitiveMeshRole, vanta_core_module

logger = logging.getLogger("VoxSigil.Security.Monitor")


@vanta_core_module(
    name="security_monitor",
    subsystem="security",
    mesh_role=CognitiveMeshRole.MONITOR,
    description="Security and compliance monitoring agent",
    capabilities=["vulnerability_scan", "permission_audit", "compliance_check"],
)
class SecurityMonitor(BaseCore):
    """
    Security monitoring agent that scans for vulnerabilities, permission issues,
    and compliance violations.
    """

    def __init__(self):
        super().__init__()
        self.last_scan_time = 0
        self.scan_interval = 3600  # 1 hour
        self.vulnerabilities = []
        self.compliance_issues = []

    async def initialize_subsystem(self, core):
        """Initialize the security monitor."""
        self.core = core
        logger.info("Security monitor initialized")

        # Start periodic scanning
        asyncio.create_task(self._periodic_scan())

    async def _periodic_scan(self):
        """Run security scans periodically."""
        while True:
            try:
                await self._run_security_scan()
                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Error in periodic security scan: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute on error

    async def _run_security_scan(self):
        """Run comprehensive security scan."""
        logger.info("Starting security scan...")

        # Run Bandit security scan
        bandit_results = await self._run_bandit_scan()

        # Check file permissions
        permission_issues = await self._check_permissions()

        # Check for compliance issues
        compliance_status = await self._check_compliance()

        # Publish results
        scan_results = {
            "timestamp": time.time(),
            "vulnerabilities": bandit_results,
            "permission_issues": permission_issues,
            "compliance": compliance_status,
            "scan_duration": time.time() - self.last_scan_time,
        }

        if hasattr(self, "core") and self.core:
            self.core.bus.publish("security.alert", scan_results)

        logger.info(f"Security scan completed. Found {len(bandit_results)} issues")

    async def _run_bandit_scan(self) -> List[Dict]:
        """Run Bandit security scanner on the codebase."""
        try:
            # Run bandit scan
            cmd = ["bandit", "-r", ".", "-f", "json", "-ll"]
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=Path.cwd()
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                results = json.loads(stdout.decode())
                return results.get("results", [])
            else:
                logger.warning(f"Bandit scan failed: {stderr.decode()}")
                return []

        except FileNotFoundError:
            logger.warning("Bandit not installed, skipping vulnerability scan")
            return []
        except Exception as e:
            logger.error(f"Error running Bandit scan: {e}")
            return []

    async def _check_permissions(self) -> List[Dict]:
        """Check for problematic file permissions."""
        issues = []

        try:
            # Check for world-writable files
            for file_path in Path(".").rglob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    # Check if world-writable (mode & 0o002)
                    if stat.st_mode & 0o002:
                        issues.append(
                            {
                                "type": "world_writable",
                                "path": str(file_path),
                                "mode": oct(stat.st_mode),
                                "severity": "medium",
                            }
                        )

            # Check for sensitive files without proper permissions
            sensitive_files = [
                "config/production_config.py",
                "config/secrets.yaml",
                ".env",
                "credentials.json",
            ]

            for file_path in sensitive_files:
                if Path(file_path).exists():
                    stat = Path(file_path).stat()
                    # Should not be readable by others
                    if stat.st_mode & 0o044:
                        issues.append(
                            {
                                "type": "sensitive_readable",
                                "path": file_path,
                                "mode": oct(stat.st_mode),
                                "severity": "high",
                            }
                        )

        except Exception as e:
            logger.error(f"Error checking permissions: {e}")

        return issues

    async def _check_compliance(self) -> Dict[str, Any]:
        """Check compliance with various standards."""
        compliance = {
            "gdpr_ready": True,
            "voice_consent": True,
            "data_retention": True,
            "issues": [],
        }

        try:
            # Check for GDPR compliance markers
            gdpr_files = ["docs/PRIVACY.md", "docs/DATA_RETENTION.md"]
            for file_path in gdpr_files:
                if not Path(file_path).exists():
                    compliance["gdpr_ready"] = False
                    compliance["issues"].append(
                        {
                            "type": "missing_gdpr_doc",
                            "description": f"Missing {file_path}",
                            "severity": "medium",
                        }
                    )

            # Check for voice consent handling
            voice_consent_files = list(Path(".").rglob("*voice*consent*"))
            if not voice_consent_files:
                compliance["voice_consent"] = False
                compliance["issues"].append(
                    {
                        "type": "missing_voice_consent",
                        "description": "No voice consent handling found",
                        "severity": "high",
                    }
                )

        except Exception as e:
            logger.error(f"Error checking compliance: {e}")

        return compliance


# UI Specification for bridge integration
def get_security_ui_spec():
    """Return UI specification for bridge integration."""
    return {
        "id": "security_monitor",
        "ui_spec": {
            "tab": "Security & Compliance",
            "widget": "SecurityPanel",
            "stream": True,
            "stream_topic": "security.alert",
            "icon": "🛡️",
        },
    }


if __name__ == "__main__":
    # Test the security monitor
    import asyncio

    async def test_security_monitor():
        monitor = SecurityMonitor()
        await monitor._run_security_scan()

    asyncio.run(test_security_monitor())
