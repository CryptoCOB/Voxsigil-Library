import json
import gzip
from datetime import datetime
from pathlib import Path
from Vanta.core.UnifiedVantaCore import get_vanta_core

class GameStateStore:
    """Simple JSON-based game state store."""

    def __init__(self, campaign_id: str = "demo"):
        self.campaign_id = campaign_id
        self.base = Path("campaigns") / campaign_id
        self.base.mkdir(parents=True, exist_ok=True)
        self.core = get_vanta_core()
        if self.core:
            self.core.register_component(
                f"vanta/tabletop/state/{campaign_id}",
                self,
                {"type": "store"},
            )

    def _state_file(self) -> Path:
        ts = datetime.now().strftime("state_%Y-%m-%d.json.gz")
        return self.base / ts

    def save(self, state: dict) -> Path:
        path = self._state_file()
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(state, f)
        return path

    def load_latest(self) -> dict | None:
        files = sorted(self.base.glob("state_*.json.gz"))
        if not files:
            return None
        with gzip.open(files[-1], "rt", encoding="utf-8") as f:
            return json.load(f)
