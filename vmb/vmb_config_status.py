#!/usr/bin/env python3
"""
VMB Configuration and Status Classes
====================================

Core configuration and status tracking classes for the VMB system.
These classes provide structured configuration management and status reporting
for the Visual Model Bootstrap system.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("VMB_Config_Status")


class VMBActivationMode(Enum):
    """VMB activation modes."""

    VMB_FIRST_RUN = "VMB_FirstRun"
    VMB_PRODUCTION = "VMB_Production"
    VMB_DEVELOPMENT = "VMB_Development"
    VMB_TESTING = "VMB_Testing"


class VMBSystemStatus(Enum):
    """VMB system status states."""

    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class VMBSwarmConfig:
    """Configuration class for VMB CopilotSwarm."""

    sigil: str = "⟠∆∇𓂀"
    agent_class: str = "CopilotSwarm"
    swarm_variant: str = "RPG_Sentinel"
    role_scope: List[str] = field(
        default_factory=lambda: ["planner", "validator", "executor", "summarizer"]
    )
    activation_mode: VMBActivationMode = VMBActivationMode.VMB_FIRST_RUN
    python_version_required: str = "3.11"
    package_manager: str = "uv"
    formatter: str = "ruff"
    max_agents: int = 4
    enable_monitoring: bool = True
    enable_recovery: bool = True
    enable_learning: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "sigil": self.sigil,
            "agent_class": self.agent_class,
            "swarm_variant": self.swarm_variant,
            "role_scope": self.role_scope,
            "activation_mode": self.activation_mode.value
            if isinstance(self.activation_mode, VMBActivationMode)
            else self.activation_mode,
            "python_version_required": self.python_version_required,
            "package_manager": self.package_manager,
            "formatter": self.formatter,
            "max_agents": self.max_agents,
            "enable_monitoring": self.enable_monitoring,
            "enable_recovery": self.enable_recovery,
            "enable_learning": self.enable_learning,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VMBSwarmConfig":
        """Create config from dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                if key == "activation_mode" and isinstance(value, str):
                    try:
                        value = VMBActivationMode(value)
                    except ValueError:
                        logger.warning(
                            f"Unknown activation mode: {value}, using default"
                        )
                        value = VMBActivationMode.VMB_FIRST_RUN
                setattr(config, key, value)
        return config


@dataclass
class VMBStatus:
    """Status tracking class for VMB system."""

    status: VMBSystemStatus = VMBSystemStatus.INACTIVE
    timestamp: datetime = field(default_factory=datetime.now)
    sigil: str = "⟠∆∇𓂀"
    agent_count: int = 0
    active_agents: List[str] = field(default_factory=list)
    system_health: Dict[str, bool] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    last_task_completed: Optional[str] = None
    uptime_seconds: float = 0.0

    def __post_init__(self):
        """Initialize default system health checks."""
        if not self.system_health:
            self.system_health = {
                "agent_health": False,
                "system_performance": False,
                "error_detection": False,
                "learning_adaptation": False,
                "threat_monitoring": False,
            }

    def update_status(self, new_status: VMBSystemStatus, message: str = ""):
        """Update system status with optional message."""
        old_status = self.status
        self.status = new_status
        self.timestamp = datetime.now()

        if message:
            self.error_log.append(
                f"{self.timestamp}: Status changed from {old_status.value} to {new_status.value} - {message}"
            )

        logger.info(f"VMB Status updated: {old_status.value} -> {new_status.value}")

    def add_active_agent(self, agent_role: str):
        """Add an active agent to the status."""
        if agent_role not in self.active_agents:
            self.active_agents.append(agent_role)
            self.agent_count = len(self.active_agents)
            logger.info(f"Agent activated: {agent_role} (Total: {self.agent_count})")

    def remove_active_agent(self, agent_role: str):
        """Remove an active agent from the status."""
        if agent_role in self.active_agents:
            self.active_agents.remove(agent_role)
            self.agent_count = len(self.active_agents)
            logger.info(f"Agent deactivated: {agent_role} (Total: {self.agent_count})")

    def update_system_health(self, component: str, healthy: bool):
        """Update health status for a system component."""
        self.system_health[component] = healthy
        status_text = "healthy" if healthy else "unhealthy"
        logger.info(f"System health updated: {component} is {status_text}")

    def is_system_healthy(self) -> bool:
        """Check if all system components are healthy."""
        return all(self.system_health.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "sigil": self.sigil,
            "agent_count": self.agent_count,
            "active_agents": self.active_agents,
            "system_health": self.system_health,
            "performance_metrics": self.performance_metrics,
            "error_log": self.error_log[-10:],  # Last 10 errors only
            "last_task_completed": self.last_task_completed,
            "uptime_seconds": self.uptime_seconds,
            "is_healthy": self.is_system_healthy(),
        }


class VMBCompletionReport:
    """Comprehensive completion report generator for VMB system."""

    def __init__(self, vmb_status: VMBStatus, config: VMBSwarmConfig):
        self.vmb_status = vmb_status
        self.config = config
        self.timestamp = datetime.now()

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive completion report."""
        report = {
            "report_info": {
                "generated": self.timestamp.isoformat(),
                "report_type": "VMB System Completion Report",
                "sigil": self.config.sigil,
            },
            "system_configuration": self.config.to_dict(),
            "current_status": self.vmb_status.to_dict(),
            "achievement_summary": self._generate_achievement_summary(),
            "performance_analysis": self._generate_performance_analysis(),
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _generate_achievement_summary(self) -> Dict[str, Any]:
        """Generate achievement summary."""
        return {
            "vmb_activation": self.vmb_status.status == VMBSystemStatus.ACTIVE,
            "agent_coordination": self.vmb_status.agent_count >= 4,
            "system_monitoring": self.vmb_status.system_health.get(
                "agent_health", False
            ),
            "error_recovery": self.vmb_status.system_health.get(
                "error_detection", False
            ),
            "learning_system": self.vmb_status.system_health.get(
                "learning_adaptation", False
            ),
            "threat_monitoring": self.vmb_status.system_health.get(
                "threat_monitoring", False
            ),
            "overall_health": self.vmb_status.is_system_healthy(),
        }

    def _generate_performance_analysis(self) -> Dict[str, Any]:
        """Generate performance analysis."""
        return {
            "uptime_hours": round(self.vmb_status.uptime_seconds / 3600, 2),
            "active_agent_count": self.vmb_status.agent_count,
            "target_agent_count": self.config.max_agents,
            "agent_utilization": f"{(self.vmb_status.agent_count / self.config.max_agents) * 100:.1f}%",
            "system_health_score": f"{sum(self.vmb_status.system_health.values()) / len(self.vmb_status.system_health) * 100:.1f}%",
            "recent_errors": len(
                [e for e in self.vmb_status.error_log if "ERROR" in e]
            ),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations."""
        recommendations = []

        if not self.vmb_status.is_system_healthy():
            recommendations.append(
                "🔧 System health check required - some components are unhealthy"
            )

        if self.vmb_status.agent_count < self.config.max_agents:
            recommendations.append(
                f"📈 Consider activating remaining agents ({self.config.max_agents - self.vmb_status.agent_count} slots available)"
            )

        if len(self.vmb_status.error_log) > 5:
            recommendations.append("⚠️ Review error log - multiple issues detected")

        if self.vmb_status.status != VMBSystemStatus.ACTIVE:
            recommendations.append(
                "🚀 System not in active state - check initialization"
            )

        if not recommendations:
            recommendations.append(
                "✅ System operating optimally - no immediate actions required"
            )

        return recommendations

    def generate_markdown_report(self) -> str:
        """Generate markdown formatted report."""
        report_data = self.generate_report()

        markdown = f"""# VMB System Completion Report
**Generated**: {report_data["report_info"]["generated"]}  
**Sigil**: {report_data["report_info"]["sigil"]}  
**Status**: {report_data["current_status"]["status"].upper()}

## Achievement Summary
"""

        for achievement, completed in report_data["achievement_summary"].items():
            status = "✅" if completed else "❌"
            markdown += f"- {status} {achievement.replace('_', ' ').title()}\n"

        markdown += f"""
## Performance Analysis
- **Uptime**: {report_data["performance_analysis"]["uptime_hours"]} hours
- **Active Agents**: {report_data["performance_analysis"]["active_agent_count"]}/{report_data["performance_analysis"]["target_agent_count"]}
- **Agent Utilization**: {report_data["performance_analysis"]["agent_utilization"]}
- **System Health Score**: {report_data["performance_analysis"]["system_health_score"]}

## Active Agents
"""

        for agent in report_data["current_status"]["active_agents"]:
            markdown += f"- 🤖 {agent.title()}\n"

        markdown += "\n## Recommendations\n"
        for rec in report_data["recommendations"]:
            markdown += f"- {rec}\n"

        return markdown

    def save_report(self, filepath: str) -> bool:
        """Save report to file."""
        try:
            markdown_report = self.generate_markdown_report()
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(markdown_report)
            logger.info(f"Report saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return False
