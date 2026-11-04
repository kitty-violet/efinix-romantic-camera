"""Same-depth video FIFO candidate: real vendor DC_FIFO, unrelated clocks."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,MS,sha,run
VIDEO=HERE.parent/'video_scaler'
DC=ROOT/'tmp/efinity-smoke-2026.1/sc431-v23-fifo2026-seed1-effort5/rtl/frame_buffer_V3/DC_FIFO.v'
def selected():return [DC,VIDEO/'guarded_async_fifo.sv',VIDEO/'framed_rgb_async_fifo.sv']
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--board',type=Path);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  records=[];sources=selected();binding=None
  if a.board:
   board=a.board.resolve();meta=json.loads((board/'inputs.json').read_text())
   mapped=json.loads((board/'result.json').read_text())
   assert sha(board/'inputs.json')==mapped['inputs_sha256']
   sources=[board/'rtl/frame_buffer_V3/DC_FIFO.v',board/'assembly/guarded_async_fifo.sv',board/'assembly/framed_rgb_async_fifo.sv']
   for f in sources:
    match=[x for x in meta['inputs'] if Path(x['snapshot']).resolve()==f]
    assert len(match)==1 and sha(f)==match[0]['sha256']
   binding=dict(board=str(board),inputs_sha256=sha(board/'inputs.json'),map_sha256=mapped['map_sha256'])
  for f in sources+[VIDEO/'tb_framed_rgb_async_fifo.sv']:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  clock_pairs=[(3.333,6.734),(2.5,3.333),(10.0,3.333)] if a.board else [(3.5,5.5)]
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__),board_binding=binding,clock_half_periods_ns=clock_pairs),indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  lines=[]
  for case,(wr,rd) in enumerate(clock_pairs):
   for seed in (71,391):
    body,_=run([MS/'vsim.exe','-c','work.tb_framed_rgb_async_fifo','-gRANDOM_SEED='+str(seed),'-gWR_HALF_PERIOD='+str(wr),'-gRD_HALF_PERIOD='+str(rd),'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,f'sim-{case}-{seed}.log',timeout=60)
    passed=re.findall(r'^# (PASS FRAMED_VIDEO_FIFO.*)$',body,re.M)
    assert len(passed)==1 and not re.search(r'\*\* (Error|Fatal)',body),body[-1600:]
    lines.append(dict(case=case,wr_half_ns=wr,rd_half_ns=rd,line=passed[0]))
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_COMPACT_VIDEO_CDC_PASS',lines=lines,board_binding=binding,inputs_sha256=sha(root/'inputs.json'),logs={f.name:sha(f) for f in root.glob('sim-*.log')},display_pipeline_connected=False,full_frame=False,pnr=False,analog_metastability_proven=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
