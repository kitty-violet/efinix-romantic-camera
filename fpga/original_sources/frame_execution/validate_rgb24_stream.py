"""Four-state full 720p byte/pixel packing, backpressure and partial reset."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import HERE,ROOT,MS,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--unpacker',type=Path);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
records=[]
for n in ('rgb24_stream_pack.sv','rgb24_stream_unpack.sv','tb_rgb24_stream.sv'):
 src=a.unpacker.resolve() if n=='rgb24_stream_unpack.sv' and a.unpacker else HERE/n
 assert src.name==n
 dst=root/n;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
(root/'inputs.json').write_text(json.dumps(dict(inputs=records),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,secs=run([MS/'vsim.exe','-c','work.tb_rgb24_stream','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=300)
lines=re.findall(r'^# (PASS RGB24_.*)$',body,re.M);assert len(lines)==2 and 'Errors: 0, Warnings: 0' in body
for r in records:assert sha(r['source'])==sha(r['snapshot'])==r['sha256']
result=dict(status='RGB24_STREAM_FOURSTATE_PASS',lines=lines,seconds=secs,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),full_system=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

