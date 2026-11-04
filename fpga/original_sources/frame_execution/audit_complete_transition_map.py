"""Audit actual emitted D cones, not just a named decode hierarchy."""
import argparse,collections,json,re
from pathlib import Path
from run_units import sha
from complete_control_transition import audit_delta,circuit
from audit_snapshot_cells import cells
from audit_local_reset_implementation import norm

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args();board=a.board.resolve()
 audit_delta(board);net=board/'outflow/ti60f225_oob.map.v';out=board/'complete_transition_map_audit.json'
 assert not out.exists();receipt=json.loads((board/'result.json').read_text());assert receipt['map_sha256']==sha(net)
 text=net.read_text();cc=cells(text);params={}
 for n,k,v in re.findall(r'^\s*defparam\s+(\S+)\s*\.(\w+)\s*=\s*([^;]+);',text,re.M):params.setdefault(norm(n),{})[k]=v.strip()
 aliases={norm(a):norm(b) for a,b in re.findall(r'^\s*assign\s+(.*?)\s*=\s*([^;]+);',text,re.M)}
 outputs={'EFX_LUT4':('O',),'EFX_COMB4':('O','P','CO'),'EFX_ADD':('O','CO'),'EFX_FF':('Q',)}
 drivers={p[o]:(n,o) for n,(k,p,_) in cc.items() for o in outputs.get(k,()) if o in p}
 local={n:(k,p) for n,(k,p,_) in cc.items() if n.startswith('cpu/ctrl_next/')}
 expected={'cpu/ctrl_next/'+n for n in circuit()[1]}
 assert set(local)==expected,(set(local)-expected,expected-set(local))
 assert all(k=='EFX_LUT4' for k,p in local.values()),local
 allcells={};states={}
 def trace(signal,found,leaves):
  if signal in ("1'b0","1'b1"):return
  if signal in aliases:return trace(aliases[signal],found,leaves)
  if signal not in drivers:leaves[signal]='boundary_or_unhandled';return
  n,o=drivers[signal];k,p,_=cc[n]
  if k=='EFX_FF':leaves[signal]=n;return
  if n in found:return
  found[n]=dict(kind=k,pins=p,params=params.get(n,{}))
  for key,value in p.items():
   if key not in outputs[k]:trace(value,found,leaves)
 for bit in range(10):
  name=f'cpu/state_oh[{bit}]~FF';kind,pins,_=cc[name]
  assert kind=='EFX_FF' and pins['CLK']=='process_clk_150' and pins['CE']=="1'b1",(name,pins)
  assert params[name]['SR_SYNC']=="1'b1" and params[name]['SR_POLARITY']=="1'b0"
  assert params[name]['SR_VALUE']==("1'b1" if bit==0 else "1'b0")
  cone,leaves={},{}
  for pin in ('D','SR','CE'):trace(pins[pin],cone,leaves)
  allcells.update(cone)
  states[str(bit)]=dict(ff=pins,params=params[name],cone=cone,leaves=leaves)
 comb={n:x for n,x in allcells.items() if x['kind']=='EFX_COMB4'}
 result=dict(status='COMPLETE_STATE_D_CONES_SINGLE_OUTPUT_PASS' if not comb else 'D_CONE_SHARED_CELL_REMAINS',
  map_sha256=sha(net),inputs_sha256=sha(board/'inputs.json'),auditor_sha256=sha(__file__),
  next_state_luts=len(local),cone_primitives=dict(collections.Counter(x['kind'] for x in allcells.values())),
  shared_cells=comb,states=states,hardware_access=False,board_release=False)
 out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('states','shared_cells')}))
 assert not comb,sorted(comb)
if __name__=='__main__':main()
