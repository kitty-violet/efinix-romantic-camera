"""Reuse bounded full-board mapper with a strict read-only DDR observation delta."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
import prepare_first_fault_board as parent
from ddr_cal_observer import BASE,MODULE,transform_apb,transform_top,transform_soc,audit_delta
PROOF=ROOT/'tmp/frame-execution/ddr-observer-short3'
def check_proof():
 r=json.loads((PROOF/'result.json').read_text());m=json.loads((PROOF/'inputs.json').read_text())
 assert r['status']=='DDR_OBSERVER_FOUR_STATE_SHORT_PASS' and r['inputs_sha256']==sha(PROOF/'inputs.json')
 assert r['log_sha256']==sha(PROOF/'simulation.log') and m['module_sha256']==sha(MODULE)
 assert m['transform_sha256']==sha(HERE/'ddr_cal_observer.py')
 for n,h in m['files'].items():assert sha(PROOF/n)==h
 j=ROOT/'tmp/frame-execution/ddr-observer-jtag2'
 jr=json.loads((j/'result.json').read_text());jm=json.loads((j/'inputs.json').read_text())
 assert jr['status']=='DDR_OBSERVER_REAL_JTAG_APB_PASS' and jr['inputs_sha256']==sha(j/'inputs.json')
 assert jr['log_sha256']==sha(j/'simulation.log') and jm['transform_sha256']==sha(HERE/'ddr_cal_observer.py')
 for n,h in jm['files'].items():assert sha(j/n)==h
 return dict(root=str(PROOF),result_sha256=sha(PROOF/'result.json'),inputs_sha256=sha(PROOF/'inputs.json'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--map-prepared',action='store_true');a=p.parse_args()
 root=a.root.resolve();proof=check_proof()
 if not a.map_prepared:
  import shutil
  from first_fault_observer import audit_delta as prior_audit
  prior_audit(BASE);assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
  for d in ('rtl','assembly'):shutil.copytree(BASE/d,root/d)
  for f in BASE.iterdir():
   if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,root/f.name)
  for rel,fn in (('rtl/ti60f225_oob_top.v',transform_top),('assembly/cnn_apb_control.sv',transform_apb),('assembly/EfxSapphireSoc.v',transform_soc)):
   (root/rel).write_text(fn((BASE/rel).read_text()),encoding='utf-8')
  meta=json.loads((BASE/'inputs.json').read_text())
  for rec in meta['inputs']:
   f=root/Path(rec['snapshot']).relative_to(BASE);rec.update(snapshot=str(f),sha256=sha(f))
  meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),ddr_observer_proof=proof,
              ddr_observer_transform_sha256=sha(HERE/'ddr_cal_observer.py'),runner_sha256=sha(__file__),hardware_access=False,board_download_allowed=False)
  (root/'inputs.json').write_text(json.dumps(meta,indent=2));(root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2))
  print('DDR_OBSERVER_PREPARED_NOT_RELEASE');return
 parent.BASE=BASE;parent.check_proof=check_proof;parent.audit_parent=lambda b:None;parent.audit_delta=audit_delta
 # The mapper itself is unchanged; status is renamed after successful completion.
 parent.main()
 result=json.loads((root/'result.json').read_text());result['status']='DDR_OBSERVER_FULL_BOARD_MAP_NOT_RELEASE'
 (root/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
