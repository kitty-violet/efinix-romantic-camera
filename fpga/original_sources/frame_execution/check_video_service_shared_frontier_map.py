"""Current interfaces and warning classification, preserving all suite checks."""
from pathlib import Path
source=Path(__file__).with_name('check_video_service_same_edge_map.py');code=source.read_text()
code=code.replace('audit_video_service_same_edge_warnings.py','audit_video_service_shared_frontier_warnings.py')
code=code.replace('validate_video_service_same_edge_pages.py','validate_video_service_shared_frontier_pages.py')
exec(compile(code,str(source)+':shared-frontier','exec'),globals())
