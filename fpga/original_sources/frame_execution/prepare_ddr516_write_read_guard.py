"""Isolated public-ROM candidate for the measured 12 CK write/read corner.

Reproduce the failing ROM first, add one NOP pair only to WR2RD, and audit
every sequence/entry relocation. No protected logic or device timing changes.
Run with the existing Efinity Python (pyrsistent is bundled there).
"""
import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
 header=(path/'ddr3_controller.vh').read_text()
 labels={k:int(v) for k,v in re.findall(r"`define (RAM_\w+) 9'd(\d+)",header)}
 return header,labels,(path/'ddr3_controller.bin').read_text().splitlines()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();project=Path('E:/fpga');assert root.is_relative_to(project/'tmp') and not root.exists()
 base=project/'tmp/frame-execution/ddr516-read-pre-guard1'
 failed=project/'tmp/frame-execution/ddr516-turnaround-rows1'
 failure=json.loads((failed/'result.json').read_text())
 assert failure['status']=='DDR516_TURNAROUND_FAIL' and failure['log_sha256']==sha(failed/'simulation.log')
 assert any('delta_ck=12 required_ck=13' in e for e in failure['errors'])
 root.mkdir();params=json.loads((base/'generation_request.json').read_text())['resolved']
 args=[]
 for k,v in params.items():
  match=re.fullmatch(r"\d+'([bdh])([0-9a-f]+)",v,re.I)
  if match:v=str(int(match[2],{'b':2,'d':10,'h':16}[match[1].lower()]))
  args.append(k+'='+v)
 original=base/'instruction_ram_read_pre_guard.py';copied=root/'instruction_ram_before.py'
 copied.write_bytes(original.read_bytes())
 def generate(script,out):
  (out/'ddr516_reference').mkdir(parents=True)
  subprocess.run([sys.executable,str(script),'--out_path',str(out),'--gen_name','ddr516_reference','--parameters',*args],check=True)
 generate(copied,root/'verify')
 for name in ('ddr3_controller.bin','ddr3_controller.vh'):
  assert sha(root/'verify/ddr516_reference'/name)==sha(failed/name),name
 # The vendor example emits sg25 for 800 MT/s. The existing strict Micron
 # fixture deliberately uses the actual device sg125 bin, independently of
 # controller operating frequency. Preserve that frozen model selection.
 device='ddr3_device_ID.vh'
 assert sha(root/'verify/ddr516_reference'/device)==sha(base/'ip/ddr516_reference'/device)
 assert (failed/device).read_text().strip()=='`define sg125'
 needle='            for i in range(latency.tW2R-6):'
 text=copied.read_text();assert text.count(needle)==1
 patched=root/'instruction_ram_write_read_guard.py'
 patched.write_text(text.replace(needle,
  '            # PROJECT: measured backpressured WR->RD is 12 CK, needs >=13.\n'
  '            # One half-rate NOP pair covers the missing CK without flags.\n'
  '            for i in range(latency.tW2R-6+2):'))
 generate(patched,root/'ip');candidate=root/'ip/ddr516_reference'
 assert sha(candidate/device)==sha(root/'verify/ddr516_reference'/device)
 (candidate/device).write_bytes((failed/device).read_bytes())
 before,old,ow=load(failed);after,new,nw=load(candidate)
 assert old.keys()==new.keys() and len(nw)==len(ow)+1
 entries=sorted(old,key=old.get);changed=[]
 for index,key in enumerate(entries):
  following=entries[index+1] if index+1<len(entries) else None
  old_segment=ow[old[key]:old[following] if following else len(ow)]
  new_segment=nw[new[key]:new[following] if following else len(nw)]
  if key=='RAM_WR2RD':
   assert new_segment==['3e00000003e0000000']+old_segment
   changed.append(key)
  else:assert new_segment==old_segment,key
  assert new[key]==old[key]+int(old[key]>old['RAM_WR2RD']),key
 # Verify all non-address header definitions, except the one-word ROM length.
 assert int(re.search(r'`define BRAM_LEN (\d+)',after)[1])==int(re.search(r'`define BRAM_LEN (\d+)',before)[1])+1
 def normalized(header):
  return re.sub(r"(`define BRAM_LEN )\d+",r'\1LENGTH',
   re.sub(r"(`define RAM_\w+ 9'd)\d+",r'\1ENTRY',header))
 assert normalized(before)==normalized(after)
 assert sha(candidate/'ddr3_device_ID.vh')==sha(failed/'ddr3_device_ID.vh')
 assert len(nw)<512
 result=dict(status='WR2RD_GUARD_STATIC_AUDIT_PASS',failed_test=str(failed),
  failed_inputs_sha256=sha(failed/'inputs.json'),failed_log_sha256=sha(failed/'simulation.log'),
  reproduced_base=True,changed_sequences=changed,added_wait_ck=2,added_wait_ns=5,
  old_words=len(ow),new_words=len(nw),old_rom_sha256=sha(failed/'ddr3_controller.bin'),
  new_rom_sha256=sha(candidate/'ddr3_controller.bin'),new_header_sha256=sha(candidate/'ddr3_controller.vh'),
  generator_before_sha256=sha(copied),generator_after_sha256=sha(patched),
  protected_logic_changed=False,device_timing_changed=False,simulation_pass=False,
  vendor_approved=False,board_release=False)
 (root/'write_read_guard.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
