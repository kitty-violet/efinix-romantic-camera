"""Check every original Gray endpoint spelling before route; no constraint edit."""
import argparse,json,re
from pathlib import Path
from run_units import sha
from video_sensor73_physical import REFERENCE
from review_input_isp_geometry import locations

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 sdc=root/'ti60f225_oob.sdc';assert sha(sdc)==sha(REFERENCE/'ti60f225_oob.sdc')
 pos=locations(root/'outflow/ti60f225_oob.out.placeloc');pins=set();buses=0
 for line in sdc.read_text().splitlines():
  if not line.startswith('set_bus_skew '):continue
  assert line.endswith(' 2.500');buses+=1
  for pin in re.findall(r'(?:protected_inst/)*u_ddr3_top/controller/[^ {}]+\|(?:CLK|D)',line):
   name,port=pin.rsplit('|',1);assert name in pos and name.endswith('~FF'),pin;pins.add(pin)
 assert buses==10 and len(pins)==140 and len([p for p in pins if p.endswith('|D')])==70
 result=dict(status='EXACT_ORIGINAL_GRAY_ENDPOINT_NAMES_PASS',buses=buses,pins=sorted(pins),destination_bits=70,missing=0,
  original_sdc_sha256=sha(sdc),renamed_endpoints=0,changed_constraints=0,routed_timing_pass=False,
  evidence={str(f):sha(f) for f in (Path(__file__),sdc,REFERENCE/'ti60f225_oob.sdc',root/'outflow/ti60f225_oob.out.placeloc')},hardware_access=False)
 (root/'original-gray-endpoints.json').write_text(json.dumps(result,indent=2));print(result['status'])

if __name__=='__main__':main()
