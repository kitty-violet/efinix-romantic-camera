"""Exact producer-scatter layout for two stride2 DW layers, no timing claim.

Each three-output-row tile uses the existing12 true-dual64-bit banks. A
24-owner batch reads one9-tap term per owner with a fixed physical port.
Replication is charged explicitly; no extra arena or fictitious read port.
"""
import argparse,collections,json
from pathlib import Path
from run_units import ROOT,sha

def source_coordinate(sw,sh,ox,oy,tap):
 ky,kx=divmod(tap,3)
 return min(sw-1,max(0,ox*2+kx-1)),min(sh-1,max(0,oy*2+ky-1))

def axis_targets(s):
 q=s//2
 if s&1:return [(q,2),(q+1,0)]
 return [(q,1)]+([(0,0)] if s==0 else [])

def location(local_x,local_y,group,tap,groups,base=0):
 return local_y*4+local_x%4,base+(local_x//4)*9*groups+tap*groups+group

def scatter(sx,sy,group,ox,oy,width,groups,base=0):
 targets=[]
 for ry,(dy,ky) in enumerate(axis_targets(sy)):
  for cx,(dx,kx) in enumerate(axis_targets(sx)):
   x,y=dx-ox,dy-oy
   if 0<=x<width and 0<=y<3:
    bank,address=location(x,y,group,3*ky+kx,groups,base)
    targets.append(dict(bank=bank,address=address,side=cx,source_candidate=ry*2+cx))
 return targets

def validate_case(sw,sh,ox,oy,width,groups,base=0):
 assert sw%2==0 and sh%2==0 and ox%8==0 and oy%3==0
 assert 0<width<=160 and width%8==0 and groups in (1,2)
 assert ox+width<=sw//2 and oy+3<=sh//2 and base+width//4*9*groups<=1024
 expected={}
 for y in range(3):
  for x in range(width):
   for g in range(groups):
    for tap in range(9):
     dest=location(x,y,g,tap,groups,base)
     assert dest not in expected
     expected[dest]=(*source_coordinate(sw,sh,ox+x,oy+y,tap),g)
 actual={};write_cycles_lower_bound=0;inputs=0;max_fanout=0
 left,top=source_coordinate(sw,sh,ox,oy,0)
 right,bottom=source_coordinate(sw,sh,ox+width-1,oy+2,8)
 for sy in range(top,bottom+1):
  for sx in range(left,right+1):
   for g in range(groups):
    inputs+=1;targets=scatter(sx,sy,g,ox,oy,width,groups,base)
    assert targets;max_fanout=max(max_fanout,len(targets))
    ports=collections.Counter()
    for t in targets:
     dest=(t['bank'],t['address']);assert dest not in actual
     actual[dest]=(sx,sy,g);ports[(t['bank'],t['side'])]+=1
    write_cycles_lower_bound+=max(ports.values())
 assert actual==expected
 # Reader mapping is fixed: owner R*8+P -> (R*4+P%4)*2+P//4.
 read_slots=0
 for b in range(width//8):
  for g in range(groups):
   for tap in range(9):
    ports=[]
    for r in range(3):
     for p in range(8):
      bank,address=location(b*8+p,r,g,tap,groups,base)
      port=bank*2+p//4;ports.append(port)
      assert expected[(bank,address)]==(*source_coordinate(sw,sh,ox+b*8+p,oy+r,tap),g)
    assert sorted(ports)==list(range(24));read_slots+=1
 return dict(sw=sw,sh=sh,ox=ox,oy=oy,width=width,groups=groups,base=base,source_words=inputs,
  scattered_words=len(actual),max_fanout=max_fanout,serialized_single_input_issue_lower_bound=write_cycles_lower_bound,
  collision_free_compute_issue_slots=read_slots,words_per_bank=width//4*9*groups,
  source_crop=[left,top,right,bottom])

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 rows=[]
 for sw,sh,g,width in [(1280,720,1,160),(640,360,2,80)]:
  for x in (0,width,sw//2-width):
   for y in (0,3,sh//2-3):
    for base in (0,360):
     rows.append(validate_case(sw,sh,x,y,width,g,base))
 assert len(rows)==36
 result=dict(status='STRIDE_TAP_SCATTER_LAYOUT_BIJECTION_AND_24_PORT_BOUND_PASS',cases=rows,
  exact_contract='replicate_1, stride2, original ky/kx/channel order; diagnostic addresses only',
  physical_banks=12,ports_per_bank=2,word_bits=64,depth=1024,extra_arena_ram=0,
  representative_tile_bytes=34560,two_buffers_bytes=69120,remaining_arena_bytes=29184,
  scatter_rtl=False,real_producer_feed=False,measured_frame_ms=None,full_frame_30fps=False,pnr=False,
  note='Issue-slot lower bounds exclude source production, scatter ingress parallelism/ACKs, RAM latency/backpressure, weights, RQ, output writes and tile transitions.',
  evidence={str(Path(__file__)):sha(__file__),str(ROOT/'network_ab/model_b_contract.json'):sha(ROOT/'network_ab/model_b_contract.json')})
 assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('cases','evidence')}))
if __name__=='__main__':main()
