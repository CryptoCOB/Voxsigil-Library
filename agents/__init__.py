"""Agent package with import fallbacks and status logging."""

from __future__ import annotations

import logging
import os
from importlib import import_module
from pathlib import Path

from .base import BaseAgent, NullAgent

logger = logging.getLogger(__name__)
_STATUS_FILE = Path(__file__).resolve().parent.parent / "agent_status.log"
_status_lines: list[str] = []


def _import_agent(module_name: str, class_name: str) -> type[BaseAgent]:
    try:
        module = import_module(f".{module_name}", __name__)
        cls = getattr(module, class_name)
        _status_lines.append(f"[OK] Registered: {class_name}")
        return cls
    except Exception as exc:  # pragma: no cover - optional agents
        logger.warning("Failed to import %s: %s", class_name, exc)
        _status_lines.append(f"[FAIL] Failed: {class_name}")
        return type(class_name, (NullAgent,), {})


Phi = _import_agent("phi", "Phi")
Voxka = _import_agent("voxka", "Voxka")
Gizmo = _import_agent("gizmo", "Gizmo")
Nix = _import_agent("nix", "Nix")
Echo = _import_agent("echo", "Echo")
Oracle = _import_agent("oracle", "Oracle")
Astra = _import_agent("astra", "Astra")
Warden = _import_agent("warden", "Warden")
Nebula = _import_agent("nebula", "Nebula")
Orion = _import_agent("orion", "Orion")
Evo = _import_agent("evo", "Evo")

OrionApprentice = _import_agent("orionapprentice", "OrionApprentice")
SocraticEngine = _import_agent("socraticengine", "SocraticEngine")
Dreamer = _import_agent("dreamer", "Dreamer")
EntropyBard = _import_agent("entropybard", "EntropyBard")
CodeWeaver = _import_agent("codeweaver", "CodeWeaver")
EchoLore = _import_agent("echolore", "EchoLore")
MirrorWarden = _import_agent("mirrorwarden", "MirrorWarden")
PulseSmith = _import_agent("pulsesmith", "PulseSmith")
BridgeFlesh = _import_agent("bridgeflesh", "BridgeFlesh")

Sam = _import_agent("sam", "Sam")
Dave = _import_agent("dave", "Dave")
Carla = _import_agent("carla", "Carla")
Andy = _import_agent("andy", "Andy")
Wendy = _import_agent("wendy", "Wendy")

VoxAgent = _import_agent("voxagent", "VoxAgent")
SDKContext = _import_agent("sdkcontext", "SDKContext")
HoloMesh = _import_agent("holomesh", "HoloMesh")

# 🧠 Codex BugPatch - Vanta Phase @2025-06-09
# SleepTimeCompute is an alias to SleepTimeComputeAgent. Having both exported
# may lead to duplicate agent entries if not handled. Kept for manifest
# compatibility.
SleepTimeComputeAgent = _import_agent("sleep_time_compute_agent", "SleepTimeComputeAgent")
SleepTimeCompute = _import_agent("sleep_time_compute_agent", "SleepTimeCompute")

if os.getenv("VANTA_DND") == "1":
    GameMasterAgent = _import_agent("game_master_agent", "GameMasterAgent")
    VoiceTableAgent = _import_agent("voice_table_agent", "VoiceTableAgent")
    RulesRefAgent = _import_agent("rules_ref_agent", "RulesRefAgent")

try:
    _STATUS_FILE.write_text("\n".join(_status_lines) + "\n", encoding="utf-8")
except Exception as exc:  # pragma: no cover - logging only
    logger.warning("Failed to write agent status: %s", exc)

__all__ = [
    "BaseAgent",
    "NullAgent",
    "Phi",
    "Voxka",
    "Gizmo",
    "Nix",
    "Echo",
    "Oracle",
    "Astra",
    "Warden",
    "Nebula",
    "Orion",
    "Evo",
    "OrionApprentice",
    "SocraticEngine",
    "Dreamer",
    "EntropyBard",
    "CodeWeaver",
    "EchoLore",
    "MirrorWarden",
    "PulseSmith",
    "BridgeFlesh",
    "Sam",
    "Dave",
    "Carla",
    "Andy",
    "Wendy",
    "VoxAgent",
    "SDKContext",
    "HoloMesh",
    "SleepTimeComputeAgent",
    "SleepTimeCompute",
]
# Include tabletop agents when enabled
if os.getenv("VANTA_DND") == "1":
    __all__ += [
        "GameMasterAgent",
        "VoiceTableAgent",
        "RulesRefAgent",
    ]
# 🧠 Codex BugPatch - Vanta Phase @2025-06-09
# Ensure agent lists do not register duplicates in UnifiedVantaCore
