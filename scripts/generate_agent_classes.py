import re
from pathlib import Path

MANIFEST = Path('AGENTS.md')
AGENT_DIR = Path('agents')

pattern = re.compile(r"^\|\s*(?P<sigil>[^|]+)\|\s*(?P<name>[^|]+)\|\s*(?P<arch>[^|]+)\|\s*(?P<class>[^|]+)\|\s*(?P<inv>[^|]+)\|\s*(?P<subs>[^|]+)\|\s*(?P<notes>[^|]+)\|")

for line in MANIFEST.read_text().splitlines():
    if line.startswith('| Sigil') or line.startswith('| ---'):
        continue
    m = pattern.match(line)
    if not m:
        continue
    d = {k: v.strip() for k, v in m.groupdict().items()}
    name = d['name'].replace(' ', '')
    file_name = f"{name.lower()}.py"
    if name == 'SleepTimeCompute':
        file_name = 'sleep_time_compute_agent.py'
    path = AGENT_DIR / file_name
    tags = [d['arch'], d['class'], d['subs'] if d['subs'] != '—' else 'None']
    invocations = [s.strip().strip('"') for s in d['inv'].split(',')]

    class_code = f'''from .base import BaseAgent


class {name}(BaseAgent):
    sigil = "{d['sigil']}"
    tags = {tags}
    invocations = {invocations}

    def initialize_subsystem(self, core):
        # Optional: bind to subsystem if defined
        pass

    def on_gui_call(self):
        # Optional: link to GUI invocation
        super().on_gui_call()

    def bind_echo_routes(self):
        # Optional: connect signals to/from UnifiedAsyncBus
        pass
'''
    path.write_text(class_code)
    print(f"Wrote {path}")
