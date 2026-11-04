"""Fixed DMA storage rewrite: independent byte oracle and measured cycle comparison."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha,run

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--variant',type=Path,required=True);p.add_argument('--reference-board',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
variant=a.variant.resolve();assert variant.is_relative_to(HERE/'variants')
board=a.reference_board.resolve();meta=json.loads((board/'inputs.json').read_text());sources={Path(r['snapshot']).name:r for r in meta['inputs']}
for name in ('frame_arena_row_dma.sv','tensor_arena12.sv'):assert sha(sources[name]['snapshot'])==sources[name]['sha256']
results=[]
for label,dma in [('reference',Path(sources['frame_arena_row_dma.sv']['snapshot'])),('candidate',variant/'frame_arena_row_dma.sv')]:
 work=root/label;work.mkdir();records=[]
 for src in (EF/'sim_models/verilog/efx_dpram10.v',Path(sources['tensor_arena12.sv']['snapshot']),dma,variant/'tb_frame_arena_row_dma.sv'):
  dst=work/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),source_sha256=sha(src),snapshot=str(dst),sha256=sha(dst)))
 (work/'inputs.json').write_text(json.dumps(dict(inputs=records),indent=2))
 run([MS/'vlib.exe','work'],work,'vlib.log')
 body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],work,'compile.log');assert 'Errors: 0, Warnings: 0' in body
 for seed in (71,391,1027):
  log=f'simulation_{seed}.log'
  body,secs=run([MS/'vsim.exe','-c','work.tb_frame_arena_row_dma',f'+SEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],work,log,timeout=90)
  line=re.findall(r'^# (PASS ROW_DMA_SHARED_ABORT.*)$',body,re.M)
  assert line==['PASS ROW_DMA_SHARED_ABORT transfers=286 failures=21 byte_checks=74150'] and 'Errors: 0, Warnings: 0' in body
  rows=[dict(zip(('store','rgb','columns','groups','cycles','error'),map(int,x))) for x in re.findall(r'^# ROW_DMA store=(\d+) rgb=(\d+) columns=(\d+) groups=(\d+) cycles=(\d+) error=(\d+)$',body,re.M)]
  results.append(dict(label=label,seed=seed,line=line[0],seconds=secs,rows=rows,inputs_sha256=sha(work/'inputs.json'),log_sha256=sha(work/log)))
  print(label,seed,line[0],flush=True)
 for r in records:assert sha(r['source'])==r['source_sha256'] and sha(r['snapshot'])==r['sha256']
result=dict(status='HELD_DMA_RETURNS_OFFICIAL_RAM_FUNCTIONAL_PASS',runs=results,byte_checks=74150*6,whole_board=False,cycle_equivalence=False,reference_board=str(board),runner_sha256=sha(__file__))
(root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'])
