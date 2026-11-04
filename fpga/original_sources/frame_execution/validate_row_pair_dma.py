"""Real RAM plus independent AXI byte model; check physical row-pair layout."""
import argparse,json,re,shutil,os
from pathlib import Path
from run_units import ROOT,HERE,MS,EF,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,action='append');p.add_argument('--routed-aux',action='store_true');p.add_argument('--dma',type=Path);p.add_argument('--testbench',type=Path);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
(r/'host-temp').mkdir();os.environ['TEMP']=os.environ['TMP']=str(r/'host-temp')
v=HERE/'variants/row_pair_arena';records=[]
for src in [EF/'sim_models/verilog/efx_dpram10.v',EF/'sim_models/verilog/efx_srl8.v',*[v/n for n in ('row_pair_arena6.sv','row_pair_dma.sv','tb_row_pair_dma.sv')],*([v/'row_pair_aux_router.sv'] if a.routed_aux else [])]:
 if a.dma and src.name=='row_pair_dma.sv':
  src=a.dma.resolve();assert src.is_relative_to(HERE/'variants') and src.name=='row_pair_dma.sv'
 if a.testbench and src.name=='tb_row_pair_dma.sv':
  src=a.testbench.resolve();assert src.is_relative_to(HERE/'variants') and src.name=='tb_row_pair_dma.sv'
 dst=r/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(src)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log');body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*(['+define+ROUTED_AUX'] if a.routed_aux else []),*[i['snapshot'] for i in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
results=[]
for seed in (a.seed or [71,391,1027]):
 body,sec=run([MS/'vsim.exe','-c','work.tb_row_pair_dma',f'+SEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,f'simulation_{seed}.log',timeout=240)
 lines=re.findall(r'^# (PASS ROW_PAIR_DMA.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
 jobs=[dict(zip(('store','rgb','columns','groups','cycles','error'),map(int,m))) for m in re.findall(r'ROW_DMA store=(\d+) rgb=(\d+) columns=(\d+) groups=(\d+) cycles=(\d+) error=(\d+)',body)]
 results.append(dict(seed=seed,line=lines[0],seconds=sec,log_sha256=sha(r/f'simulation_{seed}.log'),jobs=jobs));print({k:v for k,v in results[-1].items() if k!='jobs'},flush=True)
for i in records:assert sha(i['source'])==sha(i['snapshot'])==i['sha256']
(r/'result.json').write_text(json.dumps(dict(status='ROW_PAIR_DMA_OFFICIAL_RAM_PASS',runs=results,inputs_sha256=sha(r/'inputs.json'),routed_aux=a.routed_aux,concurrent_peer_reads=a.routed_aux,integrated_cnn=False,whole_board=False),indent=2)+'\n')
