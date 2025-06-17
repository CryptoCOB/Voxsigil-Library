"""
VoxSigil Development Mode Configuration Manager
Provides centralized configuration for all GUI tabs and components with dev mode options.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("VoxSigilDevConfig")


@dataclass
class TabConfig:
    """Configuration for a single tab."""

    enabled: bool = True
    dev_mode: bool = False
    auto_refresh: bool = False
    refresh_interval: int = 5000  # milliseconds
    debug_logging: bool = False
    show_advanced_controls: bool = False
    custom_settings: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_settings is None:
            self.custom_settings = {}


@dataclass
class NeuralTTSConfig:
    """Neural TTS specific configuration."""

    preferred_engine: str = "auto"
    voice_profile_override: Optional[str] = None
    speed_multiplier: float = 1.0
    energy_multiplier: float = 1.0
    pitch_adjustment: float = 0.0
    enable_text_enhancement: bool = True
    cache_audio_files: bool = True
    max_cache_size_mb: int = 100
    enable_voice_morphing: bool = False
    dev_show_engine_stats: bool = False
    dev_show_synthesis_time: bool = False


@dataclass
class AgentConfig:
    """Agent-specific configuration."""

    enabled: bool = True
    voice_enabled: bool = True
    auto_greet_on_startup: bool = False
    status_update_interval: int = 1000
    show_detailed_metrics: bool = False
    dev_mode_verbose: bool = False
    custom_greeting: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training tab configuration."""

    auto_start_monitoring: bool = False
    default_batch_size: int = 32
    default_learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 10
    save_checkpoints: bool = True
    checkpoint_interval: int = 10
    dev_show_gradients: bool = False
    dev_show_loss_details: bool = False
    dev_enable_profiling: bool = False


@dataclass
class VisualizationConfig:
    """Visualization tab configuration."""

    default_plot_theme: str = "dark"
    auto_update_plots: bool = True
    plot_update_interval: int = 2000
    max_data_points: int = 1000
    enable_3d_plots: bool = True
    show_grid: bool = True
    show_legends: bool = True
    dev_show_render_stats: bool = False
    dev_enable_plot_caching: bool = True


@dataclass
class PerformanceConfig:
    """Performance monitoring configuration."""

    monitor_cpu: bool = True
    monitor_memory: bool = True
    monitor_gpu: bool = True
    monitor_network: bool = False
    update_interval: int = 1000
    history_length: int = 300
    alert_cpu_threshold: float = 80.0
    alert_memory_threshold: float = 85.0
    dev_detailed_metrics: bool = False
    dev_export_metrics: bool = False


@dataclass
class MusicConfig:
    """Music tab configuration."""

    default_volume: float = 0.7
    enable_visualizations: bool = True
    auto_play_on_generation: bool = False
    default_genre: str = "ambient"
    default_tempo: int = 120
    enable_real_time_effects: bool = False
    dev_show_audio_metrics: bool = False
    dev_enable_advanced_synthesis: bool = False


@dataclass
class GridFormerConfig:
    """GridFormer tab configuration."""

    auto_start_processing: bool = False
    default_grid_size: int = 64
    visualization_enabled: bool = True
    real_time_updates: bool = True
    update_interval: int = 500
    max_iterations: int = 1000
    dev_show_internal_state: bool = False
    dev_enable_step_debugging: bool = False


class VoxSigilDevConfig:
    """
    Centralized development mode configuration manager for VoxSigil GUI.
    """

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "voxsigil_dev_config.json"
        self.config_path = Path(self.config_file)

        # Tab configurations
        self.tabs: Dict[str, TabConfig] = {}

        # Component-specific configurations
        self.neural_tts = NeuralTTSConfig()
        self.agents: Dict[str, AgentConfig] = {}
        self.training = TrainingConfig()
        self.visualization = VisualizationConfig()
        self.performance = PerformanceConfig()
        self.music = MusicConfig()
        self.gridformer = GridFormerConfig()

        # Global settings
        self.global_dev_mode = False
        self.global_debug_logging = False
        self.auto_save_config = True
        self.config_version = "1.0"

        self._initialize_default_configs()
        self.load_config()

    def _initialize_default_configs(self):
        """Initialize default configurations for all tabs."""

        # Default tab configurations
        default_tabs = [
            "models",
            "model_discovery",
            "training",
            "novel_reasoning",
            "visualization",
            "performance",
            "gridformer",
            "music",
            "neural_tts",
            "echo_log",
            "mesh_map",
            "agent_status",
            "blt_rag",
            "arc",
            "vanta_core",
        ]

        for tab_name in default_tabs:
            self.tabs[tab_name] = TabConfig()

        # Default agent configurations
        default_agents = ["Nova", "Aria", "Kai", "Echo", "Sage"]
        for agent_name in default_agents:
            self.agents[agent_name] = AgentConfig()

    def enable_dev_mode(self, component: Optional[str] = None):
        """Enable dev mode globally or for a specific component."""
        if component is None:
            self.global_dev_mode = True
            for tab_config in self.tabs.values():
                tab_config.dev_mode = True
                tab_config.show_advanced_controls = True
                tab_config.debug_logging = True
        else:
            if component in self.tabs:
                self.tabs[component].dev_mode = True
                self.tabs[component].show_advanced_controls = True
                self.tabs[component].debug_logging = True

        if self.auto_save_config:
            self.save_config()

    def disable_dev_mode(self, component: Optional[str] = None):
        """Disable dev mode globally or for a specific component."""
        if component is None:
            self.global_dev_mode = False
            for tab_config in self.tabs.values():
                tab_config.dev_mode = False
                tab_config.show_advanced_controls = False
                tab_config.debug_logging = False
        else:
            if component in self.tabs:
                self.tabs[component].dev_mode = False
                self.tabs[component].show_advanced_controls = False
                self.tabs[component].debug_logging = False

        if self.auto_save_config:
            self.save_config()

    def get_tab_config(self, tab_name: str) -> TabConfig:
        """Get configuration for a specific tab."""
        return self.tabs.get(tab_name, TabConfig())

    def update_tab_config(self, tab_name: str, **kwargs):
        """Update configuration for a specific tab."""
        if tab_name not in self.tabs:
            self.tabs[tab_name] = TabConfig()

        for key, value in kwargs.items():
            if hasattr(self.tabs[tab_name], key):
                setattr(self.tabs[tab_name], key, value)

        if self.auto_save_config:
            self.save_config()

    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_name, AgentConfig())

    def update_agent_config(self, agent_name: str, **kwargs):
        """Update configuration for a specific agent."""
        if agent_name not in self.agents:
            self.agents[agent_name] = AgentConfig()

        for key, value in kwargs.items():
            if hasattr(self.agents[agent_name], key):
                setattr(self.agents[agent_name], key, value)

        if self.auto_save_config:
            self.save_config()

    def save_config(self):
        """Save configuration to file."""
        try:
            config_data = {
                "version": self.config_version,
                "global_dev_mode": self.global_dev_mode,
                "global_debug_logging": self.global_debug_logging,
                "auto_save_config": self.auto_save_config,
                "tabs": {name: asdict(config) for name, config in self.tabs.items()},
                "neural_tts": asdict(self.neural_tts),
                "agents": {name: asdict(config) for name, config in self.agents.items()},
                "training": asdict(self.training),
                "visualization": asdict(self.visualization),
                "performance": asdict(self.performance),
                "music": asdict(self.music),
                "gridformer": asdict(self.gridformer),
            }

            with open(self.config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def load_config(self):
        """Load configuration from file."""
        if not self.config_path.exists():
            logger.info("No existing config file found, using defaults")
            return

        try:
            with open(self.config_path, "r") as f:
                config_data = json.load(f)

            # Load global settings
            self.global_dev_mode = config_data.get("global_dev_mode", False)
            self.global_debug_logging = config_data.get("global_debug_logging", False)
            self.auto_save_config = config_data.get("auto_save_config", True)

            # Load tab configurations
            if "tabs" in config_data:
                for tab_name, tab_data in config_data["tabs"].items():
                    self.tabs[tab_name] = TabConfig(**tab_data)

            # Load component configurations
            if "neural_tts" in config_data:
                self.neural_tts = NeuralTTSConfig(**config_data["neural_tts"])

            if "agents" in config_data:
                for agent_name, agent_data in config_data["agents"].items():
                    self.agents[agent_name] = AgentConfig(**agent_data)

            if "training" in config_data:
                self.training = TrainingConfig(**config_data["training"])

            if "visualization" in config_data:
                self.visualization = VisualizationConfig(**config_data["visualization"])

            if "performance" in config_data:
                self.performance = PerformanceConfig(**config_data["performance"])

            if "music" in config_data:
                self.music = MusicConfig(**config_data["music"])

            if "gridformer" in config_data:
                self.gridformer = GridFormerConfig(**config_data["gridformer"])

            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")

    def reset_to_defaults(self):
        """Reset all configurations to default values."""
        self.tabs.clear()
        self.agents.clear()
        self.neural_tts = NeuralTTSConfig()
        self.training = TrainingConfig()
        self.visualization = VisualizationConfig()
        self.performance = PerformanceConfig()
        self.music = MusicConfig()
        self.gridformer = GridFormerConfig()
        self.global_dev_mode = False
        self.global_debug_logging = False

        self._initialize_default_configs()

        if self.auto_save_config:
            self.save_config()

    def export_config(self, export_path: str):
        """Export configuration to a specific file."""
        original_path = self.config_path
        self.config_path = Path(export_path)
        self.save_config()
        self.config_path = original_path

    def import_config(self, import_path: str):
        """Import configuration from a specific file."""
        original_path = self.config_path
        self.config_path = Path(import_path)
        self.load_config()
        self.config_path = original_path

        if self.auto_save_config:
            self.save_config()


# Global configuration instance
_dev_config = None


def get_dev_config() -> VoxSigilDevConfig:
    """Get the global development configuration instance."""
    global _dev_config
    if _dev_config is None:
        _dev_config = VoxSigilDevConfig()
    return _dev_config


def is_dev_mode(component: Optional[str] = None) -> bool:
    """Check if dev mode is enabled globally or for a specific component."""
    config = get_dev_config()
    if component is None:
        return config.global_dev_mode
    else:
        tab_config = config.get_tab_config(component)
        return tab_config.dev_mode or config.global_dev_mode


# Convenience functions
def enable_global_dev_mode():
    """Enable development mode globally."""
    get_dev_config().enable_dev_mode()


def disable_global_dev_mode():
    """Disable development mode globally."""
    get_dev_config().disable_dev_mode()


def toggle_dev_mode(component: Optional[str] = None):
    """Toggle development mode on/off."""
    config = get_dev_config()
    if is_dev_mode(component):
        config.disable_dev_mode(component)
    else:
        config.enable_dev_mode(component)
