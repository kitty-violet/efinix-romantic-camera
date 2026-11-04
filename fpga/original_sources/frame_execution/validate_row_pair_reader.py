"""Compare old and new layouts using independent address-to-logical-word models."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,EF,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--official-ram',action='store_true');p.add_argument('--routed-aux',action='store_true');a=p.parse_args();assert not a.routed_aux or a.official_ram
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
# Verify deterministic case coverage before paying official RAM simulation cost.
if a.official_ram:
 selected=(0,1,2,3,4,11,47,95,143);minimum_packets=0
 for c in selected:
  width=(1,7,8,9,33,80,160,320)[(c//6)%8];paired=c%6 in (3,4);rgb=c%6==4;pw=c%6 in (0,4,5)
  if paired and not rgb:width=min(width,160)
  groups=1 if rgb else (1<<(c%2)) if paired else (1<<((c//6)%3));out_groups=4 if pw and not rgb else groups
  minimum_packets+=((width+1)//2+7 if rgb else width+8) if paired else (((width+7)//8)*(8*groups*out_groups if pw else 9)+(width-1)%8+(0 if pw else 1))
 assert minimum_packets>=7000 and sum(c%22==11 for c in selected)>=2 and 0 in selected
variant=HERE/'variants/row_pair_arena';reference=ROOT/'tmp/frame-execution/connected-board-compact-feed1/assembly/unified_column_reader.sv';records=[]
sources=[(reference,'reader_reference.sv'),(variant/'row_pair_column_reader.sv','row_pair_column_reader.sv'),(variant/'tb_row_pair_column_reader.sv','tb_row_pair_column_reader.sv')]
if a.official_ram:sources.extend([(EF/'sim_models/verilog/efx_dpram10.v','efx_dpram10.v'),(variant/'row_pair_arena6.sv','row_pair_arena6.sv')])
if a.routed_aux:sources.extend([(variant/n,n) for n in ('row_pair_aux_router.sv','row_pair_memory_subsystem.sv')])
for src,name in sources:
 dst=r/name
 if name=='reader_reference.sv':dst.write_text(src.read_text().replace('module unified_column_reader(','module reader_reference('),encoding='utf-8')
 else:shutil.copy2(src,dst)
 records.append(dict(source=str(src),snapshot=str(dst),source_sha256=sha(src),snapshot_sha256=sha(dst)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*(['+define+OFFICIAL_RAM'] if a.official_ram else []),*(['+define+ROUTED_AUX'] if a.routed_aux else []),*[p['snapshot'] for p in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
results=[]
for seed in ((71,) if a.official_ram else (71,391,1027)):
 body,seconds=run([MS/'vsim.exe','-c','work.tb_row_pair_column_reader',f'+SEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,f'simulation_{seed}.log',timeout=240 if a.official_ram else 90)
 lines=re.findall(r'^# (PASS ROW_PAIR_READER.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
 results.append(dict(seed=seed,line=lines[0],seconds=seconds,log_sha256=sha(r/f'simulation_{seed}.log')));print(results[-1],flush=True)
for p in records:assert sha(p['source'])==p['source_sha256'] and sha(p['snapshot'])==p['snapshot_sha256']
(r/'result.json').write_text(json.dumps(dict(status='ROW_PAIR_READER_LAYOUT_EQUIVALENCE_PASS',runs=results,inputs_sha256=sha(r/'inputs.json'),memory='official EFX_DPRAM10 candidate, independent reference' if a.official_ram else 'independent synchronous four-state behavioral models',official_ram_integrated=a.official_ram,routed_aux=a.routed_aux,executed_cases=9 if a.official_ram else 144,integrated_cnn=False,whole_board=False),indent=2)+'\n')
