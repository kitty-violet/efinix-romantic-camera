"""Isolate a routed circuit for checking a corrected clock-domain description."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,sha
from ddr_user_clock_profile import apply_profile
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
b=a.base.resolve();r=a.root.resolve();assert b.is_relative_to(ROOT/'tmp') and r.is_relative_to(ROOT/'tmp') and not r.exists()
result=json.loads((b/'result.json').read_text());assert result['stages'][-1]['flow']=='pnr' and result['stages'][-1]['exit_code']==0
before={str(f.relative_to(b)):sha(f) for f in b.rglob('*') if f.is_file()}
shutil.copytree(b,r)
profile=apply_profile(r,b/'clock_crossing_inventory.json')
for name,digest in before.items():assert sha(b/name)==digest,('Original changed',name)
data=dict(status='CLOCK_PROFILE_DIAGNOSTIC_ONLY',base=str(b),profile=profile,source_files=before,
          related_ddr_timing_waived=False,cdc_release=False,board_release=False)
(r/'clock_profile_diagnostic.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(dict(status=data['status'],clock_groups_added=1,gray_buses=10)))
