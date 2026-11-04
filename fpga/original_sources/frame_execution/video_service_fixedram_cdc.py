"""Actual-map banked OFP5 transport, retaining the original CDC audits."""
import argparse,json
from pathlib import Path
from run_units import sha
source=Path(__file__).with_name('snapshot_ram_cdc.py')
code=source.read_text()
changes=[
 ("if n.startswith(PREFIX) and k in ('EFX_RAM10','EFX_DPRAM10')",
  "if n.startswith('service_serializer/') and k in ('EFX_RAM10','EFX_DPRAM10')"),
 ("if n.startswith(PREFIX) and k=='EFX_FF'",
  "if n.startswith((PREFIX,'service_serializer/')) and k=='EFX_FF'"),
 ("registers[PREFIX+'scan_bank~FF']","registers['service_serializer/scan_bank~FF']"),
 ("assert len(registers)<1800", "assert len(registers)<750"),
 ("if not end.startswith(PREFIX):continue",
  "if not end.startswith((PREFIX,'service_serializer/')):continue")]
for old,new in changes:
 assert code.count(old)==1,old;code=code.replace(old,new)
exec(compile(code,str(source)+':fixedram-service','exec'),globals())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert not a.out.exists();r=structure(a.board.resolve())
 r['base_checker_sha256']=sha(source);a.out.write_text(json.dumps(r,indent=2))
 print(json.dumps({k:r[k] for k in ('status','observer_ff','duplicate_snapshot_and_shifter_removed','hardware_access')}))
