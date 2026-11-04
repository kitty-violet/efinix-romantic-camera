"""Directed faults at admission, held AW/W and partial resident RGB words."""
import argparse
import json
import shutil
from pathlib import Path
from run_units import HERE, ROOT, MS, sha, run

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
files=[HERE.parent/'video_scaler/axi_frame_write_dma.sv',HERE/'rgb24_stream_pack.sv',
       HERE/'variants/resident_rgb2/rgb720_ddr_writer.sv',HERE/'tb_rgb_resident_abort.sv']
rows=[]
for source in files:
 target=root/source.name;shutil.copy2(source,target)
 rows.append(dict(source=str(source),snapshot=str(target),sha256=sha(target)))
(root/'inputs.json').write_text(json.dumps(dict(inputs=rows,runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
log,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in rows]],root,'compile.log')
assert 'Errors: 0, Warnings: 0' in log
(root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
log,seconds=run([MS/'vsim.exe','-c','work.tb_rgb_resident_abort','-do','run.do'],root,'simulation.log',timeout=90)
assert 'PASS RGB_RESIDENT_ABORT cases=43' in log and 'Errors: 0, Warnings: 0' in log
for row in rows:assert sha(row['source'])==sha(row['snapshot'])==row['sha256']
result=dict(status='RGB_RESIDENT_ABORT_PASS',cases=43,seconds=seconds,
            inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),whole_board=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
