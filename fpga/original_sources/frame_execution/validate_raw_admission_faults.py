"""Exercise bad-token counter wrap, reset ownership and B errors on mapped inputs."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(exist_ok=False)
board=ROOT/'tmp/frame-execution/connected-board-raw-credit1'
meta=json.loads((board/'inputs.json').read_text());byname={Path(r['snapshot']).name:r for r in meta['inputs']}
names=['guarded_async_fifo.sv','raw32_frame_packer.sv','raw32_frame_cdc.sv','camera_axi_write_path.sv','axi_frame_write_dma.sv','DC_FIFO.v']
records=[]
for n in names:
 r=byname[n];src=Path(r['snapshot']);assert sha(src)==r['sha256']
 dst=root/n;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
src=HERE/'tb_raw_admission_faults.sv';dst=root/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
(root/'inputs.json').write_text(json.dumps(dict(inputs=records,map_inputs_sha256=sha(board/'inputs.json'),runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log');run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
body,seconds=run([MS/'vsim.exe','-c','work.tb_raw_admission_faults','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=180)
lines=re.findall(r'^# (PASS RAW_ADMISSION_.*)$',body,re.M);assert len(lines)==3
result=dict(status='RAW_ADMISSION_FAULTS_PASS',lines=lines,seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'))
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
