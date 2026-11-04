"""Independent complete-board map with verified finite-run native delta."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from cnn_run_budget_native import PARENT,OUT as TRANSFORM
from camera_factor_compact_board import audit_delta as parent_audit
from native_local_drivers import CELL,graph
from complete_native_logic_split import resource_counts,protected_blocks
from map_native_local_drivers import flattened_public_ports
from audit_native_roundtrip import verify
from probe_native_open_ports import compare_boundary
BOARD=ROOT/'tmp/frame-execution/connected-board-cnn-run-budget1'
CHECK=ROOT/'tmp/frame-execution/cnn-native-budget-short5/result.json'
def audit_delta(board):
 parent_audit(PARENT)
 m=json.loads((board/'inputs.json').read_text());r=json.loads((board/'result.json').read_text())
 assert r['status']=='CNN_RUN_BUDGET_FULL_BOARD_MAP_PASS'
 assert m['run_parent_sha256']==sha(PARENT/'inputs.json')
 for f,h in r['evidence'].items():assert sha(f)==h,f
 for row in m['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 assert r['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v') and r['vdb_sha256']==sha(board/'outflow/ti60f225_oob.vdb')
 return dict(status='CNN_RUN_BUDGET_FULL_VIDEO_A_MAP_BOUND',map_sha256=r['map_sha256'],hardware_access=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--map-existing',action='store_true');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
 parent_audit(PARENT);t=json.loads((TRANSFORM/'result.json').read_text());short=json.loads(CHECK.read_text())
 assert t['status']=='CNN_RUN_BUDGET_NATIVE_DELTA_PREPARED' and short['status']=='NATIVE_COUNTER_AND_CPU_CYCLE_READBACK_PASS'
 page_check=ROOT/'tmp/frame-execution/cnn-progress-page1/result.json'
 page=json.loads(page_check.read_text());assert page['status']=='NATIVE_PACKING_SCAN_HOST_DECODE_PASS'
 for f,h in page['inputs'].items():assert sha(f)==h
 for f,h in {**t['inputs'],**short['inputs']}.items():assert sha(f)==h,f
 assert sha(TRANSFORM/'native_source.v')==t['source_sha256']
 b=BOARD
 if not(a.map_existing or a.audit_only):
  assert not b.exists();b.mkdir();m=json.loads((PARENT/'inputs.json').read_text())
  for row in m['inputs']:
   src=Path(row['snapshot']);dst=b/src.relative_to(PARENT);dst.parent.mkdir(parents=True,exist_ok=True)
   shutil.copy2(TRANSFORM/'native_source.v' if dst.name=='native_source.v' else src,dst)
   row.update(snapshot=str(dst),sha256=sha(dst))
  module=HERE/'variants/run_budget/cnn_frame_admission.sv';shutil.copy2(module,b/module.name)
  m['inputs'].append(dict(source=str(module),snapshot=str(b/module.name),sha256=sha(module)))
  xml=b/'ti60f225_oob.xml';s=xml.read_text();anchor='<efx:design_file name="native_source.v" version="default" library="default" />';assert s.count(anchor)==1
  s=s.replace(anchor,anchor+'\n<efx:design_file name="cnn_frame_admission.sv" version="default" library="default" />');xml.write_text(s,encoding='utf8')
  for row in m['inputs']:row['sha256']=sha(row['snapshot'])
  m.update(run_parent_sha256=sha(PARENT/'inputs.json'),run_transform_sha256=sha(TRANSFORM/'result.json'),run_builder_sha256=sha(__file__),hardware_access=False)
  (b/'inputs.json').write_text(json.dumps(m,indent=2))
  for n in ['map.cmd','camera_graft.json','recovery_graft.json','capture_burst_graft.json']:shutil.copy2(PARENT/n,b/n)
 else:m=json.loads((b/'inputs.json').read_text())
 if a.prepare_only:print('PREPARED',b,flush=True);return
 for row in m['inputs']:assert sha(row['snapshot'])==row['sha256']
 if a.audit_only:log=(b/'map.log').read_text();sec=None
 else:
  assert not (b/'map.log').exists();log,sec=run(['cmd.exe','/d','/c','map.cmd'],b,'map.log',480)
 assert re.search(r'map\s*:\s*PASS',log)
 source=(b/'native_source.v').read_text();mapped=(b/'outflow/ti60f225_oob.map.v').read_text()
 sg,mg=graph(source),graph(mapped);missing=set(sg)-set(mg)
 assert not missing,sorted(missing)
 added=set(mg)-set(sg)
 filtered=CELL.sub(lambda z:z[0] if z[2].lstrip('\\') in sg else '',mapped)
 conn=verify(flattened_public_ports(source),flattened_public_ports(filtered));boundary=compare_boundary(source,mapped,conn['renamed_nets'])
 (b/'public_connectivity_audit.json').write_text(json.dumps(conn,indent=2))
 assert protected_blocks(source)==protected_blocks((PARENT/'outflow/ti60f225_oob.map.v').read_text())
 counts=resource_counts(b/'outflow/ti60f225_oob.map.rpt');assert counts['EFX_RAM10']+counts['EFX_DPRAM10']==235
 assert counts.get('EFX_DSP48',0)+counts.get('EFX_DSP24',0)//2==160
 assert '[EFX-0654' not in (b/'outflow/ti60f225_oob.warn.log').read_text()
 evidence={str(f):sha(f) for f in [Path(__file__),TRANSFORM/'result.json',CHECK,page_check,b/'public_connectivity_audit.json',HERE/'variants/run_budget/cnn_frame_admission.sv']}
 r=dict(status='CNN_RUN_BUDGET_FULL_BOARD_MAP_PASS',seconds=sec,counts=counts,added_cells=sorted(added),
  protected_boundary=boundary,inputs_sha256=sha(b/'inputs.json'),map_sha256=sha(b/'outflow/ti60f225_oob.map.v'),vdb_sha256=sha(b/'outflow/ti60f225_oob.vdb'),
  evidence=evidence,physical_pass=False,board_release=False,hardware_access=False)
 assert not (b/'result.json').exists();(b/'result.json').write_text(json.dumps(r,indent=2));print(r['status'],counts,flush=True)
if __name__=='__main__':main()
