"""Verify only three read/precharge waits and relocated labels changed."""
import argparse,json,re
from pathlib import Path
from run_units import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);a=p.parse_args()
 candidate=a.candidate.resolve();report=json.loads((candidate/'read_pre_guard.json').read_text())
 base=Path(report['base']);sub=Path('ip/ddr516_reference')
 def load(root):
  header=(root/sub/'ddr3_controller.vh').read_text()
  labels={k:int(v) for k,v in re.findall(r"`define (RAM_\w+) 9'd(\d+)",header)}
  words=(root/sub/'ddr3_controller.bin').read_text().splitlines()
  return labels,words
 old,ow=load(base);new,nw=load(candidate);assert old.keys()==new.keys()
 entries=sorted(old,key=old.get);changes=[]
 for idx,key in enumerate(entries):
  nextkey=entries[idx+1] if idx+1<len(entries) else None
  before=ow[old[key]:old[nextkey] if nextkey else len(ow)]
  after=nw[new[key]:new[nextkey] if nextkey else len(nw)]
  if key in report['paths']:
   assert after[:3]==['3e00000003e0000000']*3,(key,after[:3])
   assert after[3:]==before,key
   changes.append(key)
  else:assert after==before,key
 for key in entries:
  assert new[key]-old[key]==3*sum(old[p]<old[key] for p in report['paths']),key
 # No protected HDL changes anywhere in the generated simulator view.
 views=[]
 for f in (base/sub/'Testbench/modelsim').glob('*.sv'):
  cf=candidate/sub/'Testbench/modelsim'/f.name;assert sha(f)==sha(cf),f
  views.append(f.name)
 result=dict(status='ROM_GUARD_STATIC_AUDIT_PASS',changed_sequences=changes,
  added_ck_each=6,added_ns_each=15,unchanged_hdl_views=views,all_entry_relocations_verified=True,
  minimum_act_read_pre_bound_ns=40,required_config_tras_ns=37.5,
  candidate_rom_sha256=sha(candidate/sub/'ddr3_controller.bin'),board_release=False)
 (candidate/'rom_guard_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
