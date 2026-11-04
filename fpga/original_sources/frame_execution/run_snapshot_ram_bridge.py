"""Actual unchanged JTAG bridge/debugger with RAM snapshot, no synthetic host bypass."""
import argparse,json,os,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha,run
MS=Path('E:/quatra/questa_fse/win64')
BASE=ROOT/'tmp/frame-execution/connected-board-split-registered2'
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 sources=[BASE/'assembly/EfxSapphireSoc.v',HERE/'variants/fixed_snapshot_ram/jtag_bringup_snapshot.sv',
  ROOT/'tmp/frame-execution/split-registered-evidence1/rtl/tb.sv',EF/'sim_models/verilog/efx_ram10.v']
 inputs=[]
 for src in sources:
  dst=root/src.name;shutil.copy2(src,dst);inputs.append(dict(source=str(src),snapshot=str(dst),sha256=sha(src)))
 (root/'inputs.json').write_text(json.dumps(dict(inputs=inputs,runner_sha256=sha(__file__),
  board_inputs_sha256=sha(BASE/'inputs.json'),unchanged_bridge_test=True,scope='Actual bridge and debugger; synthetic APB target'),indent=2))
 try:
  run([MS/'vlib.exe','work'],root,'vlib.log',60)
  run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in inputs]],root,'compile.log',60)
  body,seconds=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',150)
  assert 'PASS COMMAND_CDC_LIVE_FIRSTFAULT_RESET commands=2' in body
  lines=re.findall(r'^# PROGRESSRAW ([0-9a-fA-F]+)$',body,re.M);assert len(lines)==2
  decoded=[]
  for line in lines:
   val=int(line,16)>>2
   assert val&0xffffffff==0x4f465035 and val>>40&255==5 and val>>32&3==1
   decoded.append(dict(magic=hex(val&0xffffffff),version=val>>40&255,sequence=val>>56&255,
     payload_hex=f'{val>>64:0240x}',valid=bool(val>>32&1),busy=bool(val>>33&1)))
  for r in inputs:assert sha(r['source'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='OFP5_RAM_REAL_BRIDGE_RESET_PROTOCOL_PASS',seconds=seconds,host_decoded_scans=decoded,
   commands=2,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),hardware_access=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as exc:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
