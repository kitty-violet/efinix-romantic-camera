"""Freeze the whole implemented board with a qualified dedicated control SoC.

Preserves all video/CNN/DDR RTL, pin assignments, clocks and constraints. The
run permit is wired into the actual pipeline enable, not a diagnostic output.
"""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,sha
from validate_fixed_control_soc import public_jtag
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--proof',type=Path,required=True)
p.add_argument('--base',type=Path,default=ROOT/'tmp/frame-execution/connected-board-fault-boundary1')
p.add_argument('--fault-proof',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();base=a.base.resolve();proof=a.proof.resolve()
assert r.is_relative_to(ROOT/'tmp') and not r.exists()
g=json.loads((proof/'result.json').read_text());assert g['status']=='PASS_FIXED_SOC_COMPLETE' and sha(proof/'inputs.json')==g['inputs_sha256']
pm=json.loads((proof/'inputs.json').read_text())
for row in pm['inputs']:assert sha(row['snapshot'])==row['sha256']
for name,h in pm['generated'].items():assert sha(proof/name)==h
fp=a.fault_proof.resolve();fg=json.loads((fp/'result.json').read_text())
assert fg['status']=='FIXED_CONTROL_FAULTS_PASS' and sha(fp/'inputs.json')==fg['inputs_sha256']
fm=json.loads((fp/'inputs.json').read_text())
for row in fm['inputs']:
 assert sha(row['snapshot'])==row['sha256'] and sha(proof/Path(row['snapshot']).name)==row['sha256']
for name,h in fm['generated'].items():assert sha(fp/name)==h
assert public_jtag(base/'assembly/EfxSapphireSoc.v')==(proof/'official_jtag.sv').read_text()
meta=json.loads((base/'inputs.json').read_text());prior=json.loads((base/'result.json').read_text());assert sha(base/'inputs.json')==prior['inputs_sha256']
for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
original_top=(base/'rtl/ti60f225_oob_top.v').read_text()
assert original_top.count('EfxSapphireSoc cpu(')==1
assert original_top.count('.enable(control_enable && cal_done_sync)')==1
top=original_top.replace('EfxSapphireSoc cpu(','wire fixed_boot_run_allowed;\ncnn_fixed_control_soc cpu(.boot_run_allowed(fixed_boot_run_allowed),')
top=top.replace('.enable(control_enable && cal_done_sync)','.enable(control_enable && cal_done_sync && fixed_boot_run_allowed)')
soc='\n'.join((proof/n).read_text() for n in ('official_jtag.sv','cnn_boot_sequencer.sv','cnn_fixed_control_soc.sv'))
r.mkdir()
for folder in ('rtl','assembly'):shutil.copytree(base/folder,r/folder)
for f in base.iterdir():
 if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,r/f.name)
records=[];changed=[]
for row in meta['inputs']:
 old=Path(row['snapshot']);rel=old.relative_to(base);dst=r/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,dst)
 if rel.as_posix()=='assembly/EfxSapphireSoc.v':dst.write_text(soc)
 if rel.as_posix()=='rtl/ti60f225_oob_top.v':dst.write_text(top)
 if sha(dst)!=row['sha256']:changed.append(rel.as_posix())
 records.append(dict(source=str(old),snapshot=str(dst),sha256=sha(dst),base_sha256=row['sha256']))
assert set(changed)=={'assembly/EfxSapphireSoc.v','rtl/ti60f225_oob_top.v'},changed
meta.update(inputs=records,base=str(base),base_inputs_sha256=sha(base/'inputs.json'),source_files=[str(r/Path(f).relative_to(base)) for f in meta['source_files']],
 fixed_control_soc=dict(proof=str(proof),result_sha256=sha(proof/'result.json'),fault_proof=str(fp),fault_result_sha256=sha(fp/'result.json'),changed_files=changed,
 host_protocol='Original serial JTAG bridge; direct aligned32-bit bounded memory requests. No RV32 injection.',
 memory_map='E1000000..E1FFFFFF DDR, F9000F00..F9000F3F mailbox, F8100000..F810006C APB read-only to JTAG',
 run_enable='control_enable AND cal_done_sync AND fixed_boot_run_allowed'),runner_sha256=sha(__file__),board_release=False)
(r/'inputs.json').write_text(json.dumps(meta,indent=2)+'\n')
for row in records:assert sha(row['snapshot'])==row['sha256']
print('FIXED_CONTROL_BOARD_FROZEN',len(records),changed)
