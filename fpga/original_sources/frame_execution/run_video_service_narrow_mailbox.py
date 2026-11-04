"""Exercise the exact generated schema5 mailbox and official RAM primitive."""
from pathlib import Path
source=Path(__file__).with_name('run_video_service_word_mailbox.py')
code=source.read_text()
changes=[
 ('ROOT,HERE,MS,sha,run','ROOT,HERE,MS,EF,sha,run'),
 ("p.add_argument('--root',type=Path,required=True);a=p.parse_args()",
  "p.add_argument('--root',type=Path,required=True);p.add_argument('--board',type=Path,required=True);a=p.parse_args();board=a.board.resolve()"),
 ("src=HERE/'variants/video_service_word/video_debug_mailbox.sv';shutil.copy2(src,out/src.name)",
  "src=board/'video_debug_mailbox.sv';shutil.copy2(src,out/src.name)\n ram=board/'video_dualclock_sample.sv';model=EF/'sim_models/verilog/efx_ram10.v'\n for f in (ram,model):shutil.copy2(f,out/f.name)"),
 ('[Path(__file__),old,src,tb,out/src.name]',
  '[Path(__file__),old,src,tb,out/src.name,ram,model,out/ram.name,out/model.name]'),
 ("[MS/'vlog.exe','-sv',tb,out/src.name]",
  "[MS/'vlog.exe','-sv','-timescale','1ns/1ps',tb,out/src.name,out/ram.name,out/model.name]"),
]
for old,new in changes:
 assert code.count(old)==1,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':schema5-generated-ram-mailbox','exec'),globals())
