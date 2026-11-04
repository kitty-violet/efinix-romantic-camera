"""Build an isolated, explicitly project-modified command-ROM candidate.

Keep protected PHY/RTL untouched. First reproduce the official ROM byte for
byte, then conservatively lengthen the three READ-to-PRE sequences. The ROM
generator recomputes every subsequent entry address. This is not a vendor fix
or a board-approved image. Run under the installed Efinity Python environment.
"""
import argparse,hashlib,json,math,os,re,shutil,subprocess,sys
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();base=a.base.resolve()
 assert root.is_relative_to(Path('E:/fpga/tmp')) and not root.exists()
 data=json.loads((base/'generation_request.json').read_text());params=data['resolved']
 assert all(params[k]==v for k,v in dict(freq='400',row='14',col='10',bank='3',data_width='1',cas_latency='6',cwl='5',DDR3_MODE='2').items())
 source=Path(os.environ['EFINITY_HOME'])/'ipm/ip/efx_ddr3_soft_controller/generator/instruction_ram_halfrate.py'
 shutil.copytree(base,root)
 shutil.copy2(source,root/'instruction_ram_original.py')
 original=source.read_text()
 args=[]
 for k,v in params.items():
  match=re.fullmatch(r"\d+'([bdh])([0-9a-f]+)",v,re.I)
  if match:v=str(int(match[2],{'b':2,'d':10,'h':16}[match[1].lower()]))
  args.append(k+'='+v)
 def generate(script,out):
  (out/'ddr516_reference').mkdir(parents=True,exist_ok=True)
  subprocess.run([sys.executable,str(script),'--out_path',str(out),'--gen_name','ddr516_reference','--parameters',*args],check=True)
 verify=root/'verify'
 generate(root/'instruction_ram_original.py',verify)
 for name in ('ddr3_controller.vh','ddr3_controller.bin','ddr3_device_ID.vh'):
  assert sha(verify/'ddr516_reference'/name)==sha(base/'ip/ddr516_reference'/name),name
 # Old path pays tRCD+tRTP, but a refresh may close a newly opened row after
 # its very first read. Pay the missing tRAS time in whole half-rate words.
 needle='        if latency.tRCD <=0 :latency.tRCD =0'
 patch='''        # PROJECT PATCH: minimum row-open time even after a single read.
        # Add only NOP pairs; preserve command/data/exit flags and regenerate
        # all ROM entry labels. No changes to PHY calibration or device limits.
        read_pre_guard_ck = 2*math.ceil(max(0, latency.tRAS
            - math.ceil(tRCD_1/cycle_ns) - tRTP_1)/2)
        latency.tRTP += read_pre_guard_ck

'''
 assert original.count(needle)==1
 patched=original.replace(needle,patch+needle)
 patched_path=root/'instruction_ram_read_pre_guard.py';patched_path.write_text(patched)
 generate(patched_path,root/'ip')
 cycle_ns=1000/int(params['freq'])
 trcd=math.ceil(float(params['tRCD'].strip('"'))/cycle_ns)
 tras=math.ceil(float(params['tRAS'].strip('"'))/cycle_ns)
 guard=2*math.ceil(max(0,tras-trcd-int(params['tRTP']))/2)
 assert guard==6
 old_words=(base/'ip/ddr516_reference/ddr3_controller.bin').read_text().splitlines()
 new_words=(root/'ip/ddr516_reference/ddr3_controller.bin').read_text().splitlines()
 assert len(new_words)-len(old_words)==9 # 3 paths * 3 half-rate words.
 report=dict(status='PROJECT_COMMAND_ROM_CANDIDATE',vendor_approved=False,board_release=False,
  original_reproduction=True,source_sha256=sha(source),patch_sha256=sha(patched_path),
  added_wait_ck=guard,added_wait_ns=guard*cycle_ns,paths=['RAM_RD2PREA','RAM_RPAW','RAM_RPAR'],
  old_words=len(old_words),new_words=len(new_words),old_rom_sha256=sha(base/'ip/ddr516_reference/ddr3_controller.bin'),
  new_rom_sha256=sha(root/'ip/ddr516_reference/ddr3_controller.bin'),base=str(base))
 data['project_patch']=report
 (root/'generation_request.json').write_text(json.dumps(data,indent=2))
 (root/'read_pre_guard.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report),flush=True)

if __name__=='__main__':main()
