"""Preserve the same 36 complete arena tiles; explicitly account for new counter CO."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('camera_factor_island_locations.py')
code=SOURCE.read_text()
needle="   assert len(a[n][1].get(p,[]))==len(b[n][1].get(p,[])),(n,p)"
replacement="""   if n=='video_processing_pipeline/pipeline/cnn/issue/add_502/i23' and p=='CO':
    # This added carry drives the two validated high counter bits. It is not
    # an arena anchor and supplies no old output alias to compare.
    assert a[n][1].get(p,[])==[]
    assert b[n][1][p]==[('net','cnn_cycles_carry24',None)]
    continue
   assert len(a[n][1].get(p,[]))==len(b[n][1].get(p,[])),(n,p)"""
code=once(code,needle,replacement)
exec(compile(code,str(SOURCE)+':run-budget-counter-carry','exec'),globals())
