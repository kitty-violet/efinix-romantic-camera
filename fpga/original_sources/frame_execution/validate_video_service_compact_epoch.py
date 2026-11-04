"""Exhaust scheduling period and verify the actual patched selector LUTs."""
import argparse,itertools,json
from pathlib import Path
from run_units import ROOT,sha
from video_service_graph import load
from video_service_compact_epoch_native import extension
from video_service_serializer_native import P,net

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 D=load();removed=set();edits={};code,r=extension(D,removed,edits,{})
 # 768 is lcm(24 page/chunk positions, 256 sequence values); cover two
 # periods, including all sequence wraps. A reset or paused clock does not
 # advance either source-side counter, so it cannot change this invariant.
 chunks={};choices=set();sweeps=[]
 for count in range(1536):
  phase=count%6;part=(count//6)%4;seq=count&255
  if phase>=2:continue
  index=part*2+phase;choice=(seq>>3)&1;progress=choice&((seq>>4)&1)
  chunks[index]=(choice,progress)
  if index==7:
   assert chunks[6]==chunks[7],(count,chunks)
   choices.add('D' if progress else 'E' if choice else 'C')
   sweeps.append(dict(last_request=count,sequence=seq,choice=choice,progress=progress));chunks={}
 assert len(sweeps)==64 and choices=={'C','E','D'}
 checks=0
 for n,pin in r['selector_loads']:
  kind,pins=D['g'][n];assert kind=='EFX_LUT4'
  ports=sorted(p for p in pins if p!='O');mask=D['ps'][n]['LUTMASK'][1]
  for bits in itertools.product((0,1),repeat=len(ports)):
   vals=dict(zip(ports,bits))
   for seq in range(32):
    # Resolve ACTUAL pins through the generated graph edits. Expected uses
    # the page contract directly, so missing/wrong edits cannot self-pass.
    actual={};expected=dict(vals)
    for port in ports:
     assert len(pins[port])==1
     original=pins[port][0];replacement=edits.get(n,{}).get(port,{}).get(original,original)
     if replacement==original:actual[port]=vals[port]
     elif replacement==net('service_page_sequence',3):actual[port]=(seq>>3)&1
     elif replacement==net('service_page_sequence',4):actual[port]=(seq>>4)&1
     else:raise AssertionError((n,port,replacement))
    expected[pin]=(seq>>3)&1
    if n=='LUT__246809':expected['I1']=(seq>>4)&1
    f=lambda v:mask>>sum(v.get('I'+str(i),0)<<i for i in range(4))&1
    assert f(actual)==f(expected);checks+=1
 out.write_text(json.dumps(dict(status='COMPACT_CAMERA_PAGE_EPOCH_AND_ISOLATION_PASS',requests=1536,sweeps=64,lut_cases=checks,
  selector_only_diagnostic=True,affected_original_capture_bits=len(r['selector_endpoints']),page_choices=sorted(choices),cnn_run_seen_independent=True,
  evidence={str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('video_service_compact_epoch_native.py')):sha(Path(__file__).with_name('video_service_compact_epoch_native.py'))},hardware_access=False,board_release=False),indent=2));print('COMPACT_CAMERA_PAGE_EPOCH_AND_ISOLATION_PASS',checks)
if __name__=='__main__':main()
