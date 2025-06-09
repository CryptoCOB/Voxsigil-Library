#!/usr/bin/env python3
"""Generate agent registry reports."""
import json
import re
from pathlib import Path
import traceback

MANIFEST_FILE = Path("AGENTS.md")
AGENT_DIR = Path("agents")

pattern = re.compile(r"^\|\s*(?P<sigil>[^|]+)\|\s*(?P<name>[^|]+)\|\s*(?P<arch>[^|]+)\|\s*(?P<class>[^|]+)\|\s*(?P<inv>[^|]+)\|\s*(?P<subs>[^|]+)\|\s*(?P<notes>[^|]+)\|")

agents = []
for line in MANIFEST_FILE.read_text().splitlines():
    if line.startswith('| Sigil') or line.startswith('| ---'):
        continue
    m = pattern.match(line)
    if m:
        d = m.groupdict()
        agents.append(
            {
                "sigil": d["sigil"].strip(),
                "name": d["name"].strip(),
                "class": d["class"].strip(),
                "invocation": d["inv"].strip(),
                "subagents": [s.strip() for s in d["subs"].split(",") if s.strip() and s.strip() != "—"],
                "notes": d["notes"].strip(),
            }
        )

available_modules = {}
for line in (AGENT_DIR / "__init__.py").read_text().splitlines():
    m = re.match(r"from \.([a-zA-Z0-9_]+) import (\w+)", line)
    if m:
        module, cls = m.groups()
        available_modules[cls] = f"agents.{module}"

log_lines = []
registered = []
try:
    from Vanta.core.UnifiedVantaCore import UnifiedVantaCore
    core = UnifiedVantaCore()
    if getattr(core, "agent_registry", None):
        registered = [name for name, _ in core.agent_registry.get_all_agents()]
except Exception as e:
    registered = []
    log_lines.append(f"UnifiedVantaCore import failed: {e}")
data = []
for entry in agents:
    cls_name = entry["name"].replace(" ", "")
    module = available_modules.get(cls_name)
    exists = bool(module)
    imported = module is not None and Path(AGENT_DIR / (module.split('.')[-1] + '.py')).exists()
    reg = cls_name in registered
    data.append({
        "name": cls_name,
        "sigil": entry["sigil"],
        "class": entry["class"],
        "invocation": entry["invocation"],
        "sub_agents": entry["subagents"],
        "status": "registered" if reg else "missing",
        "dependencies": [module] if module else [],
    })
    if not exists:
        log_lines.append(f"Module for agent {cls_name} missing")
    if not imported:
        log_lines.append(f"Agent file not importable for {cls_name}")
    if not reg:
        log_lines.append(f"Agent {cls_name} not registered in VantaCore")

Path("agents.json").write_text(json.dumps(data, indent=2))

graph_edges = []
for entry in data:
    for sa in entry.get("sub_agents", []):
        graph_edges.append({"from": entry["name"], "to": sa})
    graph_edges.append({"from": entry["name"], "to": "UnifiedVantaCore"})
Path("agent_graph.json").write_text(json.dumps(graph_edges, indent=2))

Path("agent_status.log").write_text("\n".join(log_lines))

print("Generated agents.json, agent_status.log and agent_graph.json")
