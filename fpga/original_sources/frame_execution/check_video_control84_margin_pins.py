"""Check refined84 pins and original Gray endpoints with bound setup reserve."""
import argparse,json,re
from pathlib import Path
from run_units import sha
from video_bandwidth83_physical import REFERENCE
from review_input_isp_geometry import locations

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 meta=json.loads((root/'physical_inputs.json').read_text());margin=meta['setup_margin']
 original=Path(margin['canonical_sdc']);sdc=root/'ti60f225_oob.sdc'
 assert sha(original)==margin['canonical_sdc_sha256']=='464ef739d92e3fb1d2b15018863be99636efc1cae42ca581a0526f305c4be262'
 assert margin['extra_setup_ns']==.3 and margin['extra_hold_ns']==0
 assert sha(sdc)==margin['optimization_sdc_sha256'] and sha(margin['proof'])==margin['proof_sha256']
 assert sdc.read_text()==original.read_text()+'\n# Optimization reserve only; release uses original SDC.\nset_clock_uncertainty -setup -to [get_clocks {*}] 0.300\n'
 files=[d/n for d in (REFERENCE,root) for n in ('ti60f225_oob.peri.xml','outflow/ti60f225_oob.interface.csv')]
 assert sha(files[0])==sha(files[2]);old=files[1].read_text().splitlines();new=files[3].read_text().splitlines()
 assert len(old)==len(new)
 diff=[(i,x,y) for i,(x,y) in enumerate(zip(old,new)) if x!=y]
 assert len(diff)<=1 and all(x.startswith('# Date: ') and y.startswith('# Date: ') for _,x,y in diff),diff
 pos=locations(root/'outflow/ti60f225_oob.out.placeloc');pins=set();buses=0
 for line in original.read_text().splitlines():
  if not line.startswith('set_bus_skew '):continue
  assert line.endswith(' 2.500');buses+=1
  for pin in re.findall(r'(?:protected_inst/)*u_ddr3_top/controller/[^ {}]+\|(?:CLK|D)',line):
   name,port=pin.rsplit('|',1);assert name in pos and name.endswith('~FF'),pin;pins.add(pin)
 assert buses==10 and len(pins)==140 and len([p for p in pins if p.endswith('|D')])==70
 evidence={str(f):sha(f) for f in files+[Path(__file__).resolve(),original,sdc,Path(margin['proof']),root/'physical_inputs.json',root/'outflow/ti60f225_oob.out.placeloc']}
 result=dict(status='EXACT_ORIGINAL_GRAY_ENDPOINT_NAMES_PASS',buses=buses,pins=sorted(pins),destination_bits=70,missing=0,
  original_sdc_sha256=sha(original),optimization_sdc_sha256=sha(sdc),renamed_endpoints=0,changed_constraints=0,
  extra_setup_optimization_only_ns=.3,routed_timing_pass=False,evidence=evidence,hardware_access=False)
 (root/'original-gray-endpoints.json').write_text(json.dumps(result,indent=2))
 (root/'original-periphery-output-check.json').write_text(json.dumps(dict(status='ORIGINAL_PERIPHERY_INPUT_AND_GENERATED_PINS_MATCH_EXCEPT_DATE',differences=diff,evidence=evidence,hardware_access=False),indent=2))
 print(json.dumps(dict(status=result['status'],buses=buses,pins=len(pins),periphery_match=True)))
if __name__=='__main__':main()
