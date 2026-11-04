"""Exhaustive high-region check of the retained, actual native address LUTs.

This evaluates the original LUT truth tables (two-state), with the exact
declared preview cut. It checks translation, owner isolation and write
isolation; existing adapter protocol tests cover the handshakes separately.
"""
import argparse,json,random,sys,re
from pathlib import Path
from video_service_graph import load
from run_units import sha,ROOT
from camera_joint_native import OUT

def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);a=p.parse_args()
 src=a.native.resolve();proof=json.loads((src/'graft.json').read_text());D=load();g,ps,drv=D['g'],D['ps'],D['drivers']
 native=(src/'native_source.v').read_text()
 virtual_base=int(re.search(r"assign service_virtual_region=10'h([0-9a-fA-F]+)\+",native)[1],16)
 P='video_processing_pipeline/pipeline/';F=P+'fabric/';N=lambda n,b=None:('net',n,b)
 edits={(r['cell'],r['pin'],tuple(r['from_token'])):tuple(r['to_token']) for r in proof['changed_loads']}
 checked=set();leaves=set();rng=random.Random(2031)
 def evaluate(values,preview=False):
  cache=dict(values)
  def ev(t):
   if t[0]!='net':return t[2] # expanded const tuple is (const,width,value)
   if t in cache:return cache[t]
   n=drv.get(t)
   if n is None or g[n][0]=='EFX_FF':leaves.add(t);cache[t]=0;return 0
   kind,pins=g[n];checked.add(n)
   def pin(name):
    q=pins.get(name,[('const',1,0)])[0]
    if preview:q=edits.get((n,name,q),q)
    return ev(q)
   if kind=='EFX_LUT4':v=(ps[n]['LUTMASK'][1]>>sum(pin('I'+str(i))<<i for i in range(4)))&1
   elif kind=='EFX_ADD':
    x,y,c=pin('I0'),pin('I1'),pin('CI')
    v=((x+y+c)>>1)&1 if t in pins['CO'] else (x+y+c)&1
   else:raise AssertionError((n,kind))
   cache[t]=v;return v
  return sum(ev(N('u_ddr3_top/address_bridge/n1507',b))<<b for b in range(4,32))
 def put(v,name,value,width=32):
  for b in range(width):v[N(name,b)]=(value>>b)&1
 def case(owner,address,disp,write,wa,preview=False):
  v={N('u_ddr3_top/address_bridge/n1401'):write}
  put(v,F+'shared_arbiter/read_select_owner',owner,2)
  put(v,F+'cpu128_araddr',address)
  for b in range(24,32):v[N(F+f'cpu_width_adapter/raddr[{b}]')]=(address>>b)&1
  put(v,P+'disp_araddr',disp);put(v,'u_ddr3_top/address_bridge/write_addr',wa)
  region=(disp>>22)&7
  put(v,'service_display_region',(region-3 if preview and 3<=region<=5 else region),3)
  return evaluate(v,preview)
 total=0
 for offset in [0,16,0x1000,0x2a2ff0,0x3ffff0]+[rng.randrange(0,1<<22)&~15 for _ in range(50)]:
  a0=0xe1000000+offset
  assert case(3,a0,0,0,0)==0x04000000+offset,('existing CPU mapping',hex(a0),hex(case(3,a0,0,0,0)))
  for bank in range(3):
   for result in (0,1):
    region=bank+3*result;target=(region<<22)+offset;virtual=(virtual_base+region)<<22|offset
    assert case(3,virtual,0,0,0)==target,('window translation',bank,result,offset)
    for owner in (0,1,2):
     # Other masters must ignore all CPU address changes.
     disp=(3+bank)<<22|offset
     assert case(owner,a0,disp,0,0)==case(owner,virtual,disp,0,0)
    wa=rng.getrandbits(32)&~15
    for owner in range(4):
     assert case(owner,virtual,(3+bank)<<22|offset,1,wa,True)==wa,('write affected',owner)
    total+=8
  for bank in range(3):
   assert case(2,a0,(3+bank)<<22|offset,0,0,True)==bank<<22|offset,('preview region',bank,offset)
   assert case(2,a0,bank<<22|offset,0,0,True)==bank<<22|offset,('input compare half remapped',bank)
   total+=2
 # Verify that only the ten specified READ-address D pins implement the
 # inspection binding; CPU write addresses are unchanged.
 rows=[r for r in proof['changed_loads'] if tuple(r['to_token'])[1]=='service_inspect_d']
 assert len(rows)==10 and all(r['pin']=='D' and ('araddr[' in r['cell'] or '/raddr[' in r['cell']) for r in rows)
 result=dict(status='NATIVE_ADDRESS_REGION_TRUTH_TABLE_PASS',cases=total,native_combinational_cells=len(checked),
  checks=['CPU staging unchanged','locked input/result region','other master isolation','all-owner write isolation','preview result remap','compare input region unchanged'],
  limits='Two-state combinational cut; no DDR electrical or CPU full protocol claim',hardware_access=False,board_release=False,
  inputs={str(f):sha(f) for f in [Path(__file__),src/'native_source.v',src/'graft.json']})
 (src/'address-region-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
