"""Keep complete mapped graph/parameter/port checks with proved current LUT aliases."""
from pathlib import Path
from shared_frontier_map_aliases import normalize_frontier_aliases
source=Path(__file__).with_name('audit_video_service_control_priority_map.py');code=source.read_text()
assert code.startswith('CURRENT_PROOF=')
end=code.index('\n');code="CURRENT_PROOF="+repr(' return normalize_frontier_aliases(a,b,pa)\n\n')+code[end:]
exec(compile(code,str(source)+':shared-frontier-exact-aliases','exec'),globals())
