"""Map the proved combinational factoring in a separate complete-video copy."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from camera_payload_board import BOARD as PARENT,audit_delta as parent_audit
from complete_native_logic_split import resource_counts,protected_blocks
from map_native_local_drivers import flattened_public_ports,lexical_roundtrip
from audit_native_roundtrip import verify
from probe_native_open_ports import compare_boundary
from native_local_drivers import graph,CELL
BOARD=ROOT/'tmp/frame-execution/connected-board-camera-ready1'
def normalized_source(before,after):
 a,b=graph(before),graph(after);missing=set(a)-set(b)
 assert missing=={'LUT__133061'},missing
 output=a['LUT__133061'][1]['O'][0];assert a['LUT__133061'][0]=='EFX_LUT4' and output==('net','n63388_2',None)
 assert sum(output in ts for n,(k,pins) in a.items() for p,ts in pins.items() if not(n=='LUT__133061' and p=='O'))==0
 # Exactly one wire declaration and its output pin; no top-level alias or
 # protected/public consumer can use this retired intermediate.
 assert len(re.findall(r'(?<![A-Za-z0-9_$])n63388_2(?![A-Za-z0-9_$])',before))==2
 pruned=CELL.sub(lambda m:'' if m[2].lstrip('\\') in missing else m[0],before)
 pruned=re.sub(r'^\s*defparam LUT__133061\.LUTMASK\s*=.*?;\s*$', '',pruned,flags=re.M)
 return pruned
def audit_delta(board):
 parent_audit(PARENT);m=json.loads((board/'inputs.json').read_text());r=json.loads((board/'result.json').read_text())
 assert m['ready_parent']==str(PARENT) and m['ready_parent_sha256']==sha(PARENT/'inputs.json')
 assert r['auditor_sha256']==sha(__file__)
 assert r['original_builder_sha256']==m['ready_builder_sha256']==sha(ROOT/'tmp/board-test/camera_ready_board_initial.py')
 for item in m['inputs']:assert sha(item['snapshot'])==item['sha256'],item['snapshot']
 for f,h in r['evidence'].items():assert sha(f)==h,f
 assert r['status']=='READY_CAMERA_FULL_BOARD_MAP_PASS' and r['inputs_sha256']==sha(board/'inputs.json')
 assert r['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v') and r['vdb_sha256']==sha(board/'outflow/ti60f225_oob.vdb')
 return dict(status='READY_CAMERA_FULL_VIDEO_A_MAP_BOUND',map_sha256=r['map_sha256'],hardware_access=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--shorts',type=Path,required=True);p.add_argument('--root',type=Path,default=BOARD);p.add_argument('--audit-only',action='store_true');a=p.parse_args()
 src=a.source.resolve();shorts=a.shorts.resolve();board=a.root.resolve();assert board.is_relative_to(ROOT/'tmp') and (board.exists() if a.audit_only else not board.exists());parent_audit(PARENT)
 proof=json.loads((src/'result.json').read_text());test=json.loads((shorts/'result.json').read_text());ti=json.loads((shorts/'inputs.json').read_text())
 assert proof['status']=='READY_FACTOR_GRAPH_AND_EXHAUSTIVE_BINARY_PASS' and proof['checks']==1696
 assert test['status']=='READY_FACTOR_EXHAUSTIVE_OFFICIAL_FOURSTATE_PASS' and test['checks']==1315840 and test['cuts']==7
 assert test['inputs_sha256']==sha(shorts/'inputs.json') and ti['source_result_sha256']==sha(src/'result.json')
 for f,h in proof['inputs'].items():assert sha(f)==h
 for f,h in proof['files'].items():assert sha(src/f)==h
 for f,h in ti['files'].items():assert sha(shorts/f)==h
 assert sha(shorts/'simulation.log')==test['log_sha256']
 if a.audit_only:
  assert not (board/'result.json').exists();m=json.loads((board/'inputs.json').read_text())
  for item in m['inputs']:assert sha(item['snapshot'])==item['sha256']
  assert sha(src/'native_source.v')==sha(board/'native_source.v')
 else:
  board.mkdir();m=json.loads((PARENT/'inputs.json').read_text())
  for item in m['inputs']:
   old=Path(item['snapshot']);dst=board/old.relative_to(PARENT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,dst);item['snapshot']=str(dst)
  shutil.copy2(src/'native_source.v',board/'native_source.v')
 assert re.findall(r'<efx:design_file name="([^"]+)"[^>]*/>',(board/'ti60f225_oob.xml').read_text())==['native_source.v']
 if a.audit_only:log=(board/'map.log').read_text();secs=None
 else:
  for item in m['inputs']:item['sha256']=sha(item['snapshot'])
  m.update(ready_parent=str(PARENT),ready_parent_sha256=sha(PARENT/'inputs.json'),ready_builder_sha256=sha(__file__),hardware_access=False)
  (board/'inputs.json').write_text(json.dumps(m,indent=2));shutil.copy2(PARENT/'map.cmd',board/'map.cmd')
  log,secs=run(['cmd.exe','/d','/c','map.cmd'],board,'map.log',timeout=480)
 assert re.search(r'map\s*:\s*PASS',log)
 before=(board/'native_source.v').read_text();net=board/'outflow/ti60f225_oob.map.v';after=net.read_text()
 normalized=normalized_source(before,after)
 connectivity=verify(flattened_public_ports(normalized),flattened_public_ports(after));assert connectivity['added_protected']==0
 lexical=lexical_roundtrip(normalized,after);boundary=compare_boundary(before,after,connectivity['renamed_nets'])
 assert protected_blocks(before)==protected_blocks((PARENT/'outflow/ti60f225_oob.map.v').read_text())
 assert len(protected_blocks(before))==len(protected_blocks(after))
 (board/'public_connectivity_audit.json').write_text(json.dumps(connectivity,indent=2))
 counts=resource_counts(board/'outflow/ti60f225_oob.map.rpt');expected=resource_counts(PARENT/'outflow/ti60f225_oob.map.rpt');expected['EFX_LUT4']+=proof['added_luts']-1;assert counts==expected,(counts,expected)
 assert '[EFX-0654' not in (board/'outflow/ti60f225_oob.warn.log').read_text()
 evidence={str(f):sha(f) for f in [src/'result.json',src/'cases.json',src/'definitions.json',shorts/'result.json',shorts/'inputs.json',shorts/'simulation.log',Path(__file__),ROOT/'tmp/board-test/camera_ready_board_initial.py']}
 result=dict(status='READY_CAMERA_FULL_BOARD_MAP_PASS',seconds=secs,counts=counts,map_sha256=sha(net),vdb_sha256=sha(board/'outflow/ti60f225_oob.vdb'),inputs_sha256=sha(board/'inputs.json'),lexical_audit=lexical,protected_boundary=boundary,evidence=evidence,auditor_sha256=sha(__file__),original_builder_sha256=m['ready_builder_sha256'],removed_proven_unused=['LUT__133061'],audit_only=a.audit_only,hardware_access=False,physical_pass=False,board_release=False)
 (board/'result.json').write_text(json.dumps(result,indent=2));delta=audit_delta(board);(board/'delta_audit.json').write_text(json.dumps(delta,indent=2))
 shutil.copy2(PARENT/'camera_graft.json',board/'camera_graft.json');print(json.dumps({k:v for k,v in result.items() if k not in ('evidence','lexical_audit')}))
if __name__=='__main__':main()
