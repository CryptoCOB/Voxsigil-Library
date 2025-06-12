from Vanta.core.UnifiedVantaCore import get_vanta_core

class InventoryManager:
    """Track player inventory and equipment."""

    def __init__(self):
        self.items = {}
        self.core = get_vanta_core()
        if self.core:
            self.core.register_component(
                "vanta/tabletop/inventory",
                self,
                {"type": "service", "provides": ["equip", "unequip"]},
            )

    def add(self, owner: str, item: dict):
        self.items.setdefault(owner, []).append(item)
        if self.core:
            self.core.emit_event("inventory_add", {"owner": owner, "item": item})

    def remove(self, owner: str, item_id: str):
        bag = self.items.get(owner, [])
        for i, itm in enumerate(bag):
            if itm.get("id") == item_id:
                bag.pop(i)
                break
        if self.core:
            self.core.emit_event("inventory_remove", {"owner": owner, "item": item_id})
