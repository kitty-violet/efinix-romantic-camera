"""Reproduce RAW/RGB ownership starvation and check the resident-credit fix."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(exist_ok=False)
 frozen=ROOT/'tmp/frame-execution/board-full-write-buffer-ram1'
 common=['guarded_async_fifo.sv','raw32_frame_packer.sv','raw32_frame_cdc.sv','camera_axi_write_path.sv',
         'axi_frame_write_dma.sv','axi_three_master_burst_arbiter.sv','DC_FIFO.v','rgb24_stream_pack.sv','rgb720_ddr_writer.sv']
 changed={'guarded_async_fifo.sv','raw32_frame_cdc.sv','camera_axi_write_path.sv'}
 results=[]
 for fixed in (False,True):
  build=root/('fixed' if fixed else 'original');build.mkdir()
  records=[]
  for n in common:
   src=HERE.parent/'video_scaler'/n if fixed and n in changed else next(frozen.rglob(n))
   dst=build/n;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
  for src in [HERE/'tb_capture_write_contention.sv',HERE.parent/'video_scaler/tb_camera_axi_write_path.sv']:
   dst=build/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
  (build/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
  run([MS/'vlib.exe','work'],build,'vlib.log')
  run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],build,'compile.log')
  body,seconds=run([MS/'vsim.exe','-c','work.tb_capture_write_contention','-gEXPECT_CREDIT='+str(int(fixed)),
       '-do','onerror {quit -code 1}; run -all; quit -code 0'],build,'contention.log',timeout=180)
  lines=re.findall(r'^# (PASS CAPTURE_CONTENTION.*)$',body,re.M);assert len(lines)==1
  if fixed:
   fault_body,_=run([MS/'vsim.exe','-c','work.tb_camera_axi_write_path','-do',
       'onerror {quit -code 1}; run -all; quit -code 0'],build,'short_frame.log')
   assert 'PASS: camera RAW32 CDC to error-safe AXI write path' in fault_body
  results.append(dict(credit=fixed,line=lines[0],seconds=seconds,inputs_sha256=sha(build/'inputs.json')))
 result=dict(status='CAPTURE_ADMISSION_AB_PASS',results=results,whole_board=False,ddr_phy=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
