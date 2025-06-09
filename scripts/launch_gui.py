#!/usr/bin/env python3
"""VoxSigil GUI launcher.

Provides a thin wrapper around the legacy VMB GUI integration so
README examples referencing ``scripts/launch_gui.py`` succeed.
"""

# 🧠 Codex BugPatch - Vanta Phase @2025-06-09
import asyncio
from legacy_gui.vmb_gui_launcher import main

if __name__ == "__main__":
    asyncio.run(main())
