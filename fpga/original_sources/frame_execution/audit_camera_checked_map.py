"""Audit exact camera graft including dead-cone removal by official synthesis."""
import collections,json,re
from pathlib import Path
from run_units import HERE,sha
from camera_checked_board import BOARD,PARENT,UNIT,audit_delta
from native_local_drivers import CELL,graph
from map_native_local_drivers import flattened_public_ports,lexical_roundtrip
from audit_native_roundtrip import verify
from probe_native_open_ports import compare_boundary
from complete_native_logic_split import resource_counts,protected_blocks

def main():
 board=BOARD;source=(board/'native_source.v').read_text();mapped=(board/'outflow/ti60f225_oob.map.v').read_text()
 assert re.search(r'map\s*:\s*PASS',(board/'map.log').read_text())
 a,b=graph(source),graph(mapped);removed=set(a)-set(b);added=set(b)-set(a)
 graft=json.loads((board/'camera_graft.json').read_text());cut=set(graft['changed_pins']);assert len(cut)==113
 assert len(removed)==663 and len(added)==988
 assert all(n.startswith(('camera_checked/','camera_observer/','camera_diagnostic[','camera_config_diagnostic[','camera_config_','camera_new_')) or re.fullmatch(r'(?:LUT__\d+|ADD__\d+|SUB__\d+)',n) for n in added)
 outputs={'EFX_LUT4':('O',),'EFX_FF':('Q',),'EFX_ADD':('O','CO'),'EFX_RAM10':('RDATA',)}
 consumers=collections.defaultdict(set)
 for n,(kind,pins) in a.items():
  for p,values in pins.items():
   if p not in outputs.get(kind,()):
    for t in values:consumers[t].add((n,p))
 # Every removed primitive is in the now unobserved old camera/probe cone.
 dead_edges=0
 for n in removed:
  kind,pins=a[n];assert kind in outputs,(n,kind)
  for p in outputs[kind]:
   for t in pins.get(p,[]):
    outside={n2 for n2,p2 in consumers[t] if n2 not in removed}
    assert not outside,('removed cell has retained consumers',n,p,sorted(outside)[:8])
    dead_edges+=len(consumers[t])
 common=set(a)&set(b)-cut
 def keep(t,names):return CELL.sub(lambda m:m[0] if m[2].lstrip('\\') in names else '',t)
 original_common=keep(source,common);mapped_common=keep(mapped,common)
 comparison=verify(flattened_public_ports(original_common),flattened_public_ports(mapped_common))
 lexical=lexical_roundtrip(original_common,mapped_common)
 boundary=compare_boundary(source,mapped,comparison['renamed_nets'])
 assert protected_blocks(source)==protected_blocks((PARENT/'native_source.v').read_text())
 # Rewired sinks stay the same FF/LUT instance; only the explicitly listed
 # input pin changes. Validate their other pins through the proven renaming.
 from audit_snapshot_cells import cells
 from probe_full_native_split_map import params
 aa,bb=cells(flattened_public_ports(source)),cells(flattened_public_ports(mapped))
 ren=comparison['renamed_nets'];ap,bp=params(source),params(mapped)
 changes={}
 for n in sorted(cut):
  changed_pin=graft['changed_pins'][n];ka,pa,_=aa[n];kb,pb,_=bb[n];assert ka==kb and ap[n]==bp[n]
  for p,v in pa.items():
   if p==changed_pin:continue
   from audit_native_roundtrip import tokens
   assert [ren.get(t,t) for t in tokens(v)]==tokens(pb[p]),(n,p,v,pb[p])
  changes[n]=dict(pin=changed_pin,before=pa[changed_pin],after=pb[changed_pin])
 # New logic is forbidden to drive any retained old sink outside the cut.
 driver_tokens=set()
 for n in added:
  kind,pins=b[n];assert kind in outputs,(n,kind)
  for p in outputs[kind]:driver_tokens.update(pins.get(p,[]))
 for n in set(b)-added-cut:
  kind,pins=b[n]
  for p,values in pins.items():
   if p in outputs.get(kind,()):continue
   assert not driver_tokens.intersection(values),('new feedback into old logic',n,p)
 comparison.update(dead_old_cells=len(removed),dead_edges=dead_edges,removed_kinds=dict(collections.Counter(a[n][0] for n in removed)),
   removed_cells=sorted(removed),explicit_changed_sinks=changes,new_cells=sorted(added),camera_has_no_control_feedback=True)
 (board/'public_connectivity_audit.json').write_text(json.dumps(comparison,indent=2))
 result=dict(status='CAMERA_CHECKED_FULL_BOARD_MAP_PASS',map_sha256=sha(board/'outflow/ti60f225_oob.map.v'),
  vdb_sha256=sha(board/'outflow/ti60f225_oob.vdb'),resources=resource_counts(board/'outflow/ti60f225_oob.map.rpt'),
  inputs_sha256=sha(board/'inputs.json'),new_cells=sorted(added),lexical_audit=lexical,protected_boundary=boundary,
  evidence={str(f):sha(f) for f in (UNIT/'result.json',board/'inputs.json',board/'camera_graft.json',board/'public_connectivity_audit.json',HERE/'camera_checked_board.py',Path(__file__))},
  post_map_audit_repaired=True,physical_pass=False,board_download_allowed=False,hardware_access=False)
 (board/'result.json').write_text(json.dumps(result,indent=2));(board/'delta_audit.json').write_text(json.dumps(audit_delta(board),indent=2))
 print(result['status'],result['resources'])
if __name__=='__main__':main()
