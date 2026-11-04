"""Official RAM four-state row bridge test: packing, boundary, stalls and errors."""
import argparse,json,re,shutil
from run_units import HERE,ROOT,EF,MS,run,sha
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wide',action='store_true');p.add_argument('--production',action='store_true');p.add_argument('--integrated',action='store_true');p.add_argument('--overrides-dir',type=Path);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
    paths=[EF/'sim_models/verilog/efx_dpram10.v',HERE/'tensor_arena12.sv',HERE/('variants/production_banked/frame_arena_row_dma.sv' if a.production else 'variants/throughput/frame_arena_row_dma.sv' if a.wide else 'frame_arena_row_dma.sv'),HERE/'tb_frame_arena_row_dma.sv']
    if a.integrated:paths=[HERE/'variants/integrated_datapath'/f.name if f.name in ('frame_arena_row_dma.sv','tb_frame_arena_row_dma.sv') else f for f in paths]
    if a.overrides_dir:
        override=a.overrides_dir.resolve();assert override.is_relative_to(HERE/'variants')
        paths=[override/f.name if (override/f.name).is_file() else f for f in paths]
    records=[]
    for f in paths:
        dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
    (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(Path(__file__))),indent=2))
    run([MS/'vlib.exe','work'],root,'vlib.log')
    body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
    body,seconds=run([MS/'vsim.exe','-c','work.tb_frame_arena_row_dma','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log')
    passed=re.findall(r'^# (PASS ROW_DMA.*)$',body,re.M);assert len(passed)==1 and 'Errors: 0, Warnings: 0' in body
    for r in records:assert sha(r['path'])==r['sha256']==sha(r['snapshot'])
    result=dict(status='ROW_DMA_OFFICIAL_RAM_SOURCE_PASS',line=passed[0],seconds=seconds,inputs_sha256=sha(root/'inputs.json'),simulation_sha256=sha(root/'simulation.log'),full_frame=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
