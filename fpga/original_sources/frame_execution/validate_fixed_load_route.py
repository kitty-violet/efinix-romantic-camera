"""Fixed-load wiring: all control cycles exact; payload compared on every valid write."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,EF,sha,run


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--reference-board',type=Path,default=ROOT/'tmp/frame-execution/connected-board-raw-credit1')
 p.add_argument('--overrides-dir',type=Path,default=HERE/'variants/shared_word_route');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 reference_board=a.reference_board.resolve();variant=a.overrides_dir.resolve()
 assert reference_board.is_relative_to(ROOT/'tmp') and variant.is_relative_to(HERE/'variants')
 reference=reference_board/'assembly/frame_arena_row_dma.sv'
 meta=json.loads((reference.parent.parent/'inputs.json').read_text())
 row=next(r for r in meta['inputs'] if Path(r['snapshot'])==reference)
 assert sha(reference)==row['sha256']
 outputs={**dict.fromkeys(('cmd_ready','done_valid','error','busy','arvalid','awvalid','rready','wlast','wvalid','bready'),1),
  **dict.fromkeys(('av','aw','sr'),24),**dict.fromkeys(('araddr','awaddr','cycles','arena_words','axi_beats'),32),
  'aa':240,'ad':1536,'at':384,'arlen':8,'awlen':8,'wdata':128,'wstrb':16}
 declarations='\n'.join(f'wire [{w-1}:0] ref_{n};' for n,w in outputs.items())
 instance='frame_arena_row_dma_reference #(.ABORT_SHARED_ARENA(1)) ref_dut('+','.join(f'.{n}(ref_{n})' for n in outputs)+',.*);'
 checks='integer equivalent_edges=0; always @(negedge clk)begin #1; equivalent_edges=equivalent_edges+1;\n'+''.join(
  f'if({n}!==ref_{n})$fatal(1,"DMA equivalence {n} at %0t",$time);\n' for n in outputs if n!='ad')+'''for(integer port=0;port<24;port=port+1)begin
 if(av[port] && aw[port] && ad[port*64+:64]!==ref_ad[port*64+:64])
  $fatal(1,"DMA write payload equivalence port=%0d at %0t",port,$time);
end
if(dut.allocation && !dut.store && (dut.rgb ? dut.issued[1:0]!=0:dut.issued[0]!=0))
 $fatal(1,"fixed load parcel alignment");
end
'''
 tb=variant/'tb_frame_arena_row_dma.sv'
 if not tb.is_file():tb=HERE/'variants/bounded_dma/tb_frame_arena_row_dma.sv'
 arena=Path(next(r['snapshot'] for r in meta['inputs'] if Path(r['snapshot']).name=='tensor_arena12.sv'))
 assert sha(arena)==next(r['sha256'] for r in meta['inputs'] if Path(r['snapshot'])==arena)
 sources=[EF/'sim_models/verilog/efx_dpram10.v',arena,variant/'frame_arena_row_dma.sv',reference,tb];records=[]
 for src in sources:
  name='frame_arena_row_dma_reference.sv' if src==reference else src.name
  dst=root/name;body=src.read_text()
  if src==reference:body=body.replace('module frame_arena_row_dma #','module frame_arena_row_dma_reference #',1)
  if src.name=='tb_frame_arena_row_dma.sv':
   marker='frame_arena_row_dma #(.ABORT_SHARED_ARENA(1)) dut(.*);'
   assert body.count(marker)==1;body=body.replace(marker,marker+'\n'+declarations+'\n'+instance+'\n'+checks)
   body=body.replace('$display("PASS ROW_DMA_SHARED_ABORT',
     '$display("DMA_EQUIVALENT_EDGES %0d",equivalent_edges);$display("PASS ROW_DMA_SHARED_ABORT')
  if src.name=='tb_frame_arena_row_dma.sv':
   body=body.replace('repeat(4)@(negedge clk);reset=0;', 'if($value$plusargs("SEED=%d",rng))begin end\n  repeat(4)@(negedge clk);reset=0;')
  dst.write_text(body);records.append(dict(source=str(src),source_sha256=sha(src),snapshot=str(dst),sha256=sha(dst)))
 (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
 run([MS/'vlib.exe','work'],root,'vlib.log')
 body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
 assert 'Errors: 0, Warnings: 0' in body
 results=[]
 for seed in (71,391,1027):
  body,seconds=run([MS/'vsim.exe','-c','work.tb_frame_arena_row_dma',f'+SEED={seed}','-do',
  'onerror {quit -code 1}; run -all; quit -code 0'],root,f'simulation_{seed}.log')
  assert 'Errors: 0, Warnings: 0' in body and 'PASS ROW_DMA_SHARED_ABORT' in body
  edges=int(re.search(r'DMA_EQUIVALENT_EDGES (\d+)',body)[1]);assert edges>17000
  results.append(dict(seed=seed,edges=edges,seconds=seconds,log_sha256=sha(root/f'simulation_{seed}.log')))
 for row in records:assert sha(row['source'])==row['source_sha256'] and sha(row['snapshot'])==row['sha256']
 result=dict(status='FIXED_LOAD_ROUTE_FOUR_STATE_CYCLE_EQUIVALENCE_PASS',runs=results,outputs=len(outputs),write_payload_only=True,
  reference_board=str(reference_board),reference_sha256=sha(reference),candidate_sha256=sha(variant/'frame_arena_row_dma.sv'),
  inputs_sha256=sha(root/'inputs.json'),whole_board=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
