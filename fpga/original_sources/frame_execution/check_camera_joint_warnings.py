"""Explain every new mapper dangling net without suppressing any warning."""
import argparse,collections,json,re
from pathlib import Path
from camera_joint_board import BOARD,PARENT
from native_local_drivers import graph
from camera_joint_native import OUT
from run_units import sha
from probe_native_open_ports import protected_pins
from audit_native_roundtrip import tokens
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists()
 current=BOARD/'outflow/ti60f225_oob.warn.log';previous=PARENT/'outflow/ti60f225_oob.warn.log'
 body=current.read_text();before=previous.read_text();source=(BOARD/'native_source.v').read_text();g=graph(source)
 assert '[EFX-0654' not in body
 names=re.findall(r"\[EFX-0201 WARNING\] Re-wiring to GND non-driven net/port '([^']+)'",body)
 samples=tokens(protected_pins(source)['u_ddr3_top/samples']);assert len(samples)==40 and len(names)==10
 bits={name:39-samples.index(name) for name in names}
 assert sorted(bits.values())==[12,13,14,15,32,33,36,37,38,39]
 assert all(len(re.findall(r'\b'+name+r'\b',source))==1 for name in names)
 for marker in ('[EFX-0200 WARNING]','[EFX-0201 WARNING]'):assert body.count(marker)==before.count(marker)
 dangling=lambda t:set(re.findall(r'Net (.*?) is dangling',t))
 added=dangling(body)-dangling(before);drivers={};loads=collections.Counter()
 for n,(k,pins) in g.items():
  for pin,ts in pins.items():
   for t in ts:
    if pin in OUT.get(k,[]):drivers[t]=(n,k,pin)
    else:loads[t]+=1
 explained=[]
 for name in sorted(added):
  t=('net',name,None);n,k,pin=drivers[t]
  assert n.startswith('JOINT_LOCAL_') and k=='EFX_ADD' and pin=='O' and loads[t]==0,(name,n,k,pin,loads[t])
  assert g[n][1].get('CO') and loads[g[n][1]['CO'][0]]==1,(n,'missing dedicated carry consumer')
  explained.append(dict(net=name,cell=n,reason='Intermediate sum unused; whole carry prefix retained to drive the selected address bit. No unused functional input.'))
 assert len(explained)==21,len(explained)
 result=dict(status='JOINT_UNUSED_CARRY_SUMS_AND_PROTECTED_OPEN_BITS_PROVED',map_sha256=sha(BOARD/'outflow/ti60f225_oob.map.v'),sample_bits=bits,new_dangling=explained,
  evidence={str(f):sha(f) for f in (current,previous,BOARD/'native_source.v',BOARD/'result.json',Path(__file__))},hardware_access=False)
 a.out.write_text(json.dumps(result,indent=2));print(result['status'],len(explained))
if __name__=='__main__':main()
