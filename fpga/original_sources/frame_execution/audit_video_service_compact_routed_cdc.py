"""Classify compact JTAG separately; retain every ordinary service crossing."""
import json
from run_units import sha
from audit_video_service_compact_jtag import structure,parsed
from audit_video_service_routed_cdc import inventory as service_inventory

def inventory(root,structure_path):
 s=json.loads(structure_path.read_text());text=(root/'outflow/ti60f225_oob.map.v').read_text();c,_=parsed(text);compact=structure(text)
 assert s['map_sha256']==sha(root/'outflow/ti60f225_oob.map.v')
 expected={(x['start'].removesuffix('|CLK'),x['end']) for x in compact['pairs']}
 ordinary=[];seen=set()
 for row in s['cross_domain_cones']:
  pair=(row['start'],row['end'])
  if pair in expected:
   assert row['source_clocks']==[[['net','process_clk_150',None]]] and row['destination_clock']==[['net','jtag_inst1_TCK',None]]
   assert pair not in seen;seen.add(pair)
  else:ordinary.append(row)
 assert seen==expected
 ports={'jtag_inst1_SEL','jtag_inst1_SHIFT','jtag_inst1_TDI'};external=[];jtag=[]
 for row in s['external_inputs']:
  if row['source'] not in ports:external.append(row);continue
  end=row['end'];n,pin=end.rsplit('|',1)
  assert pin in ('D','CE') and c[n][0]=='EFX_FF' and c[n][1]['CLK']=='jtag_inst1_TCK'
  assert n=='service_serial_tdo~FF' or n.startswith(('service_serializer/shifter[','service_serializer/dff_11/i8_rst_'))
  assert row['clock']==[['net','jtag_inst1_TCK',None]]
  jtag.append(row)
 assert {r['source'] for r in jtag}==ports
 assert len(jtag)==(1352 if compact.get('dualclock_sample_ram') else 1274)
 filtered=dict(s,cross_domain_cones=ordinary,external_inputs=external)
 path=root/'service_non_jtag_inventory.json';assert not path.exists();path.write_text(json.dumps(filtered,indent=2))
 result=service_inventory(root,path)
 result.update(compact_inventory_sha256=sha(structure_path),compact_jtag_pairs=compact['pairs'],synchronous_jtag_input_cones=len(jtag),compact_jtag_routed_settling_required=True,split_checker_sha256=sha(__file__))
 return result
