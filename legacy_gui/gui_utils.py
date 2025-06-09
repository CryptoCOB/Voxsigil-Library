import json
import tkinter as tk
from typing import Optional


class _ToolTip:
    """Simple tooltip for Tkinter widgets."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.root = widget.winfo_toplevel()
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
        self.root.bind("<Destroy>", self.hide)

    def show(self, _event=None) -> None:
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.geometry(f"+{x}+{y}")
        tk.Label(
            tw,
            text=self.text,
            bg="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Consolas", 8),
        ).pack()

    def hide(self, _event=None) -> None:
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def bind_agent_buttons(root: tk.Misc, registry, meta_path: str = "agents.json") -> None:
    """Bind agent on_gui_call buttons to a Tkinter root or frame."""
    if not root or not registry:
        return

    meta: dict[str, dict] = {}
    try:
        with open(meta_path, "r") as f:
            for entry in json.load(f):
                meta[entry.get("name")] = entry
    except Exception:
        meta = {}

    frame = tk.Frame(root, bg="#2a2a4e")
    frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    tk.Label(frame, text="Agents", bg="#2a2a4e", fg="#00ff88", font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=5)

    for agent_name, agent in registry.get_all_agents():
        if not hasattr(agent, "on_gui_call"):
            continue

        meta_entry = meta.get(agent_name, {})
        invocations = getattr(agent, "invocations", [])
        label = invocations[0] if invocations else f"Invoke {agent_name}"

        btn = tk.Button(
            frame,
            text=label,
            command=lambda a=agent: a.on_gui_call(),
            bg="#4ecdc4",
            fg="white",
            font=("Consolas", 9, "bold"),
            relief=tk.RAISED,
            bd=1,
        )
        tooltip = f"{meta_entry.get('class', '')} | {', '.join(getattr(agent, 'tags', []))}"
        _ToolTip(btn, tooltip)
        btn.pack(side=tk.LEFT, padx=2)
