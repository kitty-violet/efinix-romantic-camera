"""Same full truth-table missing-port audit for the owner comparison."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_mapped_boot_warnings.py');code=source.read_text()
for old,new in [('VIDEO_SERVICE_ACTUAL_MAP_BOOT_PASS','ACTUAL_MAP_OWNER_FAILURE_BOUNDARY_PASS'),('mapped_boot','mapped_owner'),("'efx_ff.v','efx_ram10.v'","'efx_ff.v'"),('root/\'result.json\',Path(__file__)','root/\'result.json\',Path(__file__),source')]:
 assert old in code,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':owner-warning-proof','exec'),globals())
