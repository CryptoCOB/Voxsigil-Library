import tkinter as tk
from typing import Optional


def bind_agent_buttons(root: tk.Misc, registry) -> None:
    """Bind agent on_gui_call buttons to a Tkinter root or frame."""
    if not root or not registry:
        return

    frame = tk.Frame(root, bg="#2a2a4e")
    frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    tk.Label(frame, text="Agents", bg="#2a2a4e", fg="#00ff88", font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=5)

    for agent_name, agent in registry.get_all_agents():
        if not hasattr(agent, "on_gui_call"):
            continue
        btn = tk.Button(
            frame,
            text=f"Invoke {agent_name}",
            command=lambda a=agent: a.on_gui_call(),
            bg="#4ecdc4",
            fg="white",
            font=("Consolas", 9, "bold"),
            relief=tk.RAISED,
            bd=1,
        )
        btn.pack(side=tk.LEFT, padx=2)
