"""Bounded fixed weight-ring and partial-sum gate using official SRLs."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=HERE/'variants/fixed_window_dot8x32';files=[v/n for n in ('fixed_ring_weights9.sv','fixed_partial32.sv','tb_fixed_phase_units.sv')]+[EF/'sim_models/verilog/efx_srl8.v'];records=[]
import os
if os.environ.get('FPGA_FIXED_OVERLAY'):
 overlay={f.name:f for f in Path(os.environ['FPGA_FIXED_OVERLAY']).resolve().glob('*.sv')}
 assert overlay and set(overlay)<=set(f.name for f in files)
 files=[overlay.get(f.name,f) for f in files]
for f in files:
 dst=r/f.name;shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(f)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log');body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,seconds=run([MS/'vsim.exe','-c','work.tb_fixed_phase_units','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=90)
lines=re.findall(r'^# (PASS FIXED_PHASE_UNITS.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
for x in records:assert sha(x['snapshot'])==sha(x['source'])==x['sha256']
result=dict(status='FIXED_PHASE_UNITS_OFFICIAL_SRL_PASS',line=lines[0],seconds=seconds,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),integrated_array=False)
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
