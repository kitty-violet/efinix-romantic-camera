"""Compare the actual specialized combinational window guard to its wide original."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=HERE/'variants/fixed_resident_header_test';module=v/'fixed_dma_window_guard.sv'
production=HERE/'variants/fixed_short_guard/cnn_frame_datapath.sv'
assert production.read_text().endswith(module.read_text())
records=[]
for f in (module,v/'tb_fixed_dma_window_guard.sv'):
 d=r/f.name;shutil.copy2(f,d);records.append(dict(source=str(f),snapshot=str(d),sha256=sha(f)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,production_sha256=sha(production),runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log')
body,_=run([MS/'vlog.exe','-sv',*[x['snapshot'] for x in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,secs=run([MS/'vsim.exe','-c','work.tb_fixed_dma_window_guard','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=90)
lines=re.findall(r'^# (PASS FIXED_DMA_WINDOW .*)$',body,re.M);assert len(lines)==1
for row in records:assert sha(row['source'])==sha(row['snapshot'])==row['sha256']
result=dict(status='FIXED_DMA_WINDOW_EQUIVALENCE_PASS',line=lines[0],seconds=secs,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'))
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
