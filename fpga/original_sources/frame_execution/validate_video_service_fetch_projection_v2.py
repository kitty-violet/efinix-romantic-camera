"""Current mapped projection binding uses retained drivers, not LUT input pins."""
from pathlib import Path
from bind_video_service_mapped_cut import bind_cut
source=Path(__file__).with_name('validate_video_service_fetch_projection.py');code=source.read_text()
a=code.index(' if side and board:');b=code.index(' assert external<=set(binds)',a)
code=code[:a]+''' if side and board:
  binds,binding_rows=bind_cut(source,new,cuts,external)
''' +code[b:]
code=code.replace("paths=[Path(__file__),", "paths=[Path(__file__),Path(__file__).with_name('validate_video_service_fetch_projection.py'),Path(__file__).with_name('bind_video_service_mapped_cut.py'),")
exec(compile(code,str(source)+':retained-drivers-v2','exec'),globals())
