"""Verify all eight expected-value cones and the unchanged ROM address stay in CLK_5M."""
import argparse,json,pickle
from pathlib import Path
from run_units import ROOT,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args();b=a.board.resolve()
 proof=json.loads((b/'sensor75-map-audit.json').read_text());assert proof['status']=='SENSOR75_EXACT_SOURCE_SCOPE_FRESH_MAP_PASS'
 for f,h in proof['evidence'].items():assert sha(f)==h
 h,g,ps=pickle.loads((b/'sensor75-map-parse.pkl').read_bytes());assert h==proof['map_sha256']==sha(b/'outflow/ti60f225_oob.map.v')
 outs={'EFX_LUT4':['O'],'EFX_FF':['Q'],'EFX_ADD':['O','CO'],'EFX_RAM10':['RDATA']}
 drivers={t:n for n,(k,pins) in g.items() for p in outs.get(k,[]) for t in pins.get(p,[])}
 endpoints=[]
 def check(name,pin):
  k,pins=g[name];clock=pins['CLK' if k=='EFX_FF' else 'RCLK'];assert clock==[('net','CLK_5M',None)]
  todo=list(pins[pin]);seen=set();origins={}
  while todo:
   t=todo.pop()
   if t[0]=='literal' or t in seen:continue
   seen.add(t);assert t in drivers,('unclassified_input',name,pin,t)
   n=drivers[t];kind,q=g[n]
   if kind in ('EFX_FF','EFX_RAM10'):
    clk=q['CLK' if kind=='EFX_FF' else 'RCLK'];assert clk==[('net','CLK_5M',None)],(n,clk);origins[n]=clk
   else:
    assert kind in ('EFX_LUT4','EFX_ADD'),(n,kind)
    todo.extend(v for p,ts in q.items() if p not in outs[kind] for v in ts)
  endpoints.append(dict(destination=name,port=pin,clock='CLK_5M',sequential_origins=origins))
 for bit in range(8):check(f'camera_config_diagnostic[{16+bit}]~FF','D')
 check('camera_checked/config_rom/ROM__D$2','RADDR')
 parent=Path(json.loads((b/'inputs.json').read_text())['parent']);source=parent/'service-cdc-inventory-expanded.json';inv=json.loads(source.read_text())
 assert inv['map_sha256']==sha(parent/'outflow/ti60f225_oob.map.v') and not inv['unclassified'];inv['map_sha256']=h
 evidence={str(f):sha(f) for f in (Path(__file__),source,b/'sensor75-map-audit.json',b/'inputs.json')}
 inv['classification_evidence'].update(evidence)
 inv['sensor75_reuse_scope']='Exact original source retains existing boundaries. All eight expected-data cones and unchanged ROM address have only CLK_5M origins. 91 outputless primitives removed, including seven unused FFs; all surviving boundaries unchanged. Zero new FF/clock; routed timing is not reused.'
 (b/'service-cdc-inventory-expanded.json').write_text(json.dumps(inv,indent=2))
 result=dict(status='SENSOR75_CDC_DELTA_CLASSIFIED_NOT_ROUTED_PASS',map_sha256=h,local_changed_endpoints=endpoints,new_clocks=0,new_sequential_cells=0,new_cross_domain_cones=0,
  current_inventory_sha256=sha(b/'service-cdc-inventory-expanded.json'),reused_service_inventory_sha256=sha(source),evidence=evidence,hardware_access=False,board_release=False)
 (b/'sensor75-cdc-delta.json').write_text(json.dumps(result,indent=2));print(result['status'])

if __name__=='__main__':main()
