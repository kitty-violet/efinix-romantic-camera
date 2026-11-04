"""Real RGB DMA/packing/display endpoints, full 720p and fault checks."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import HERE,ROOT,MS,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--burst',type=int,default=128);p.add_argument('--read-burst',type=int,default=16);p.add_argument('--row-stall',type=int,default=0)
p.add_argument('--writer',type=Path);p.add_argument('--reader',type=Path);p.add_argument('--unpacker',type=Path);p.add_argument('--write-base',type=lambda x:int(x,0),default=0x00400000);p.add_argument('--resident-check',action='store_true');a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
paths=[HERE/n for n in ('rgb24_stream_pack.sv','rgb24_stream_unpack.sv','rgb720_ddr_writer.sv','rgb720_ddr_reader.sv','tb_rgb720_ddr.sv')]
for i,override in ((1,a.unpacker),(3,a.reader)):
 if override:
  assert override.name==paths[i].name
  paths[i]=override.resolve()
if a.writer:
 assert a.writer.name=='rgb720_ddr_writer.sv'
 paths[2]=a.writer.resolve()
paths.insert(0,HERE.parent/'video_scaler/axi_frame_write_dma.sv')
records=[]
for src in paths:
 dst=root/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
(root/'inputs.json').write_text(json.dumps(dict(inputs=records,capture_burst=a.burst,display_burst=a.read_burst,row_stall=a.row_stall,write_base=a.write_base,resident_check=a.resident_check,runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
(root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
body,secs=run([MS/'vsim.exe','-c','work.tb_rgb720_ddr','-gCAPTURE_BURST='+str(a.burst),'-gDISPLAY_BURST='+str(a.read_burst),'-gROW_STALL='+str(a.row_stall),'-gWRITE_BASE='+str(a.write_base),'-gCHECK_RESIDENT='+str(int(a.resident_check)),'-do','run.do'],root,'simulation.log',timeout=600)
lines=re.findall(r'^# (PASS RGB720_.*)$',body,re.M);assert len(lines)==9 and 'Errors: 0, Warnings: 0' in body
for r in records:assert sha(r['source'])==sha(r['snapshot'])==r['sha256']
result=dict(status='RGB720_DDR_FOURSTATE_PASS',capture_burst=a.burst,display_burst=a.read_burst,row_stall=a.row_stall,write_base=a.write_base,resident_check=a.resident_check,lines=lines,seconds=secs,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),full_system=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

