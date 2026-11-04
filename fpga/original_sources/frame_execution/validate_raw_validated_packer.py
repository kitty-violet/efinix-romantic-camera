"""Independent short frame-verdict gate; not the new full camera bridge."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
files=[HERE/'variants/raw_stream/raw32_validated_packer.sv',HERE/'variants/raw_stream_test/tb_raw32_validated_packer.sv']
if a.source:
 assert a.source.resolve().is_relative_to(HERE/'variants') and a.source.name==files[0].name
 files[0]=a.source.resolve()
rows=[]
for f in files:
 d=r/f.name;shutil.copy2(f,d);rows.append(dict(source=str(f),snapshot=str(d),sha256=sha(d)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=rows,runner_sha256=sha(__file__)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log');b,_=run([MS/'vlog.exe','-sv',*[x['snapshot'] for x in rows]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in b
b,t=run([MS/'vsim.exe','-c','work.tb_raw32_validated_packer','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=30)
m=re.findall(r'^# (PASS RAW_VALIDATED_PACKER.*)$',b,re.M);assert len(m)==1 and 'Errors: 0, Warnings: 0' in b
for x in rows:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
result=dict(status='RAW_VALIDATED_PACKER_PASS',line=m[0],seconds=t,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),cdc_proved=False,whole_camera_chain_proved=False)
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
