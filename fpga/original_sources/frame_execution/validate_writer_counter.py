"""Cyclic writer equivalence, invalid mapping rejection and full program applicability."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=HERE/'variants/fixed_writer_counter';ref=HERE/'variants/fixed_indexed_payload/fixed_column_store4.sv'
records=[]
for src in (v/'fixed_column_store4.sv',v/'tb_writer_counter.sv',ref):
 dst=r/('reference_writer.sv' if src==ref else src.name)
 if src==ref:dst.write_text(src.read_text().replace('module fixed_column_store4(', 'module reference_writer('))
 else:shutil.copy2(src,dst)
 records.append(dict(source=str(src),snapshot=str(dst),source_sha256=sha(src),sha256=sha(dst)))
program=ROOT/'tmp/frame-execution/paired-ddr-program1/program.json'
commands=[c for c in json.loads(program.read_text()) if c['kind']=='compute']
assert len(commands)==8789
for c in commands:
 for i,row in enumerate(c['outputs']):assert row['bank']<6,c['command']
meta=dict(inputs=records,program_sha256=sha(program),compute_commands=len(commands),runner_sha256=sha(__file__))
(r/'inputs.json').write_text(json.dumps(meta,indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,seconds=run([MS/'vsim.exe','-c','work.tb_writer_counter','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=90)
line=re.findall(r'^# (PASS WRITER_COUNTER .*)$',body,re.M)
assert len(line)==1 and not re.search(r'\*\* (Fatal|Error)',body),body[-2000:]
for f in records:assert sha(f['source'])==f['source_sha256'] and sha(f['snapshot'])==f['sha256']
result=dict(status='WRITER_COUNTER_REFERENCE_AND_PROGRAM_PASS',line=line[0],seconds=seconds,compute_commands=8789,
 inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'))
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
