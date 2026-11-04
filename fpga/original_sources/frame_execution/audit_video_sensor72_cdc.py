"""Reuse classified crossings only after proving the added logic is CLK_5M-only."""
import argparse,json,pickle
from pathlib import Path
from run_units import ROOT,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args();board=a.board.resolve()
 proof=json.loads((board/'sensor72-map-audit.json').read_text());assert proof['status']=='SENSOR72_ALL_RETAINED_MAP_PASS'
 h,g,ps=pickle.loads((board/'sensor72-map-parse.pkl').read_bytes());assert h==proof['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
 outputs={'EFX_LUT4':['O'],'EFX_FF':['Q'],'EFX_ADD':['O','CO'],'EFX_RAM10':['RDATA']}
 drivers={t:n for n,(k,pins) in g.items() for pin in outputs.get(k,[]) for t in pins.get(pin,[])}
 endpoints=[]
 for bit in range(1,5):
  name=f'camera_config_diagnostic[{16+bit}]~FF';kind,pins=g[name]
  assert kind=='EFX_FF' and pins['CLK']==[('net','CLK_5M',None)]
  todo=list(pins['D']);seen=set();origins={}
  while todo:
   t=todo.pop()
   if t[0]=='literal' or t in seen:continue
   seen.add(t);assert t in drivers,('new_async_input',name,t)
   n=drivers[t];k,q=g[n]
   if k in ('EFX_FF','EFX_RAM10'):
    clk=q['CLK' if k=='EFX_FF' else 'RCLK'];assert clk==[('net','CLK_5M',None)],(name,n,clk)
    origins[n]=clk
   else:
    assert k in ('EFX_LUT4','EFX_ADD'),(n,k)
    todo.extend(u for pin,ts in q.items() if pin not in outputs[k] for u in ts)
  endpoints.append(dict(destination=name,clock='CLK_5M',sequential_origins=origins))
 parent=Path(json.loads((board/'inputs.json').read_text())['parent'])
 source=parent/'service-cdc-inventory-expanded.json';inventory=json.loads(source.read_text())
 assert inventory['map_sha256']==sha(parent/'outflow/ti60f225_oob.map.v') and not inventory['unclassified']
 inventory['map_sha256']=h
 evidence={str(f):sha(f) for f in (Path(__file__),source,board/'sensor72-map-audit.json',board/'inputs.json')}
 inventory['classification_evidence'].update(evidence)
 inventory['sensor72_reuse_scope']='All existing mapped cells and crossing boundaries equivalent except four CLK_5M-local D cones; new same-domain origins checked explicitly; routed timing not reused.'
 (board/'service-cdc-inventory-expanded.json').write_text(json.dumps(inventory,indent=2))
 result=dict(status='SENSOR72_CDC_DELTA_CLASSIFIED_NOT_ROUTED_PASS',map_sha256=h,local_changed_endpoints=endpoints,
  new_clocks=0,new_sequential_cells=0,new_cross_domain_cones=0,reused_service_inventory_sha256=sha(source),
  current_inventory_sha256=sha(board/'service-cdc-inventory-expanded.json'),evidence=evidence,hardware_access=False,board_release=False)
 (board/'sensor72-cdc-delta.json').write_text(json.dumps(result,indent=2));print(result['status'],flush=True)

if __name__=='__main__':main()
