import secrets

__all__ = ["resolve_attack", "roll_dice", "d20"]

def roll_dice(expr: str) -> int:
    """Roll dice using a simple NdM+K expression."""
    total = 0
    parts = expr.lower().replace(' ', '').split('+')
    for part in parts:
        if 'd' in part:
            num, sides = part.split('d')
            num = int(num or 1)
            sides = int(sides)
            for _ in range(num):
                total += secrets.randbelow(sides) + 1
        else:
            total += int(part)
    return total

def d20(adv: bool = False) -> int:
    """Roll a d20 with optional advantage."""
    if not adv:
        return secrets.randbelow(20) + 1
    return max(secrets.randbelow(20) + 1, secrets.randbelow(20) + 1)

def resolve_attack(attacker, target, weapon, adv: bool = False):
    """Resolve a basic D&D style attack."""
    roll = d20(adv) + attacker.mod("STR") + attacker.prof
    if roll >= target.AC:
        dmg = roll_dice(weapon.dmg) + attacker.mod("STR")
        return True, dmg
    return False, 0
