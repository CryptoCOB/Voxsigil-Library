import secrets
from Vanta.core.UnifiedVantaCore import get_vanta_core

class DiceRollerService:
    """Cryptographically secure dice roller."""

    def __init__(self):
        self.core = get_vanta_core()
        if self.core:
            self.core.register_component(
                "vanta/tabletop/dice_roller",
                self,
                {"type": "service", "provides": ["dice.roll"]},
            )

    def roll(self, expr: str) -> int:
        total = 0
        parts = expr.lower().replace(" ", "").split("+")
        for part in parts:
            if "d" in part:
                num, sides = part.split("d")
                num = int(num or 1)
                sides = int(sides)
                for _ in range(num):
                    total += secrets.randbelow(sides) + 1
            else:
                total += int(part)
        if self.core:
            self.core.emit_event("dice_roll", {"expr": expr, "total": total})
        return total
