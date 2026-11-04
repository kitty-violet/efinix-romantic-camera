"""Fixed 24-port stride2 feed + real 128-MAC/32-RQ pool; scatter producer remains separate."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,MS,sha,run
from validate_stride_tap_scatter import vectors
from validate_mixed_pool import selected as mixed_selected
from validate_compute_mailbox import selected as all_selected
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--lockstep',action='store_true');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  oracle=vectors(root/'vectors');words=[]
  from stride_tap_layout import source_coordinate
  def signed(v):return (v&255)-256 if v&128 else v&255
  for ci,(sw,sh,ox,oy,w,g,base) in enumerate(oracle['shapes']):
   for r in range(3):
    for x in range(w):
     for group in range(g):
      out=0
      for lane in range(8):
       value=0
       for tap in range(9):
        sx,sy=source_coordinate(sw,sh,ox+x,oy+r,tap)
        v=signed(sx*37+sy*71+group*19+lane*23+ci*11)
        value+=v*((tap*3+lane*2)%7-3)
       q=(abs(value)+16)//32
       if value<0:q=-q
       out|=(max(-128,min(127,q))&255)<<(8*lane)
      words.append(out)
  (root/'vectors/result.hex').write_text(''.join(f'{x:016x}\n' for x in words))
  wanted={p.name for p in mixed_selected()}
  paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_ram10.v','efx_dpram10.v')]+[p for p in all_selected() if p.name in wanted]+[HERE/n for n in ('tensor_arena12.sv','stride_tap_feed24.sv','tb_stride_tap_feed24.sv')]
  if a.lockstep:paths=[HERE/'variants/stride_lockstep'/f.name if f.name=='stride_tap_feed24.sv' else f for f in paths]
  records=[]
  for f in paths:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  m=dict(inputs=records,runner_sha256=sha(__file__),oracle=oracle,vectors={f.name:sha(f) for f in (root/'vectors').iterdir()})
  (root/'inputs.json').write_text(json.dumps(m,indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  body,secs=run([MS/'vsim.exe','-c','work.tb_stride_tap_feed24','+VECTORS='+(root/'vectors').as_posix(),'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log')
  lines=re.findall(r'^# (PASS STRIDE_TAP_FEED24.*)$',body,re.M);assert len(lines)==1
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_STRIDE24_REAL_RAM_MAC_PASS',line=lines[0],rows=re.findall(r'^# TAP_FEED (.*)$',body,re.M),seconds=secs,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),producer_connected=False,weights_prefetched_by_tb=True,result_writer_connected=False,full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()

