"""Tabletop module registration for Vanta-DM."""
import os
from Vanta.core.UnifiedVantaCore import get_vanta_core
from agents import GameMasterAgent, VoiceTableAgent, RulesRefAgent

components = {}

if os.getenv("VANTA_DND") == "1":
    core = get_vanta_core()
    gm = GameMasterAgent(vanta_core=core)
    vt = VoiceTableAgent(vanta_core=core)
    ref = RulesRefAgent(vanta_core=core)
    components = {
        "game_master": gm,
        "voice_table": vt,
        "rules_ref": ref,
    }
    for name, comp in components.items():
        if core:
            core.register_component(f"vanta/tabletop/{name}", comp)

