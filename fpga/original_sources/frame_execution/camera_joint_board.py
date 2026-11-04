"""Independent full-board reimport of the proved joint combinational delta."""
import argparse,json,re,shutil,traceback,collections
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from camera_capture_prefetch_board import BOARD as PARENT,audit_delta as parent_audit
from complete_native_logic_split import resource_counts,protected_blocks
from map_native_local_drivers import flattened_public_ports,lexical_roundtrip
from audit_native_roundtrip import verify
from probe_native_open_ports import compare_boundary
from native_local_drivers import CELL,graph
BOARD=ROOT/'tmp/frame-execution/connected-board-camera-joint3'
TRANSFORM=ROOT/'tmp/frame-execution/camera-joint-native6'

def transform_check():
 r=json.loads((TRANSFORM/'result.json').read_text())
 assert r['status']=='JOINT_NATIVE_EXACT_TRANSFORM_PASS'
 assert r['divider_exhaustive_patterns']==8192 and r['cycles_added']==0
 assert r['source_sha256']==sha(TRANSFORM/'native_source.v')
 for f,h in r['inputs'].items():assert sha(f)==h,f
 return r

def audit_delta(board):
 parent_audit(PARENT);transform_check()
 m=json.loads((board/'inputs.json').read_text());r=json.loads((board/'result.json').read_text())
 assert r['status']=='CAMERA_JOINT_FULL_BOARD_MAP_PASS'
 assert m['joint_parent_sha256']==sha(PARENT/'inputs.json')
 assert m['joint_transform_sha256']==sha(TRANSFORM/'result.json') and m['joint_builder_sha256']==sha(__file__)
 for row in m['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 for f,h in r['evidence'].items():assert sha(f)==h,f
 assert sha(board/'outflow/ti60f225_oob.map.v')==r['map_sha256']
 assert sha(board/'outflow/ti60f225_oob.vdb')==r['vdb_sha256']
 return dict(status='CAMERA_JOINT_FULL_VIDEO_A_MAP_BOUND',map_sha256=r['map_sha256'],hardware_access=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--map-existing',action='store_true');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
 b=BOARD;proof=transform_check();parent_audit(PARENT)
 if not (a.map_existing or a.audit_only):
  assert not b.exists();b.mkdir();m=json.loads((PARENT/'inputs.json').read_text())
  for row in m['inputs']:
   src=Path(row['snapshot']);dest=b/src.relative_to(PARENT);dest.parent.mkdir(parents=True,exist_ok=True)
   shutil.copy2(TRANSFORM/'native_source.v' if dest.name=='native_source.v' else src,dest)
   row.update(snapshot=str(dest),sha256=sha(dest))
  # These modules have already been mapped into native_source; do not list
  # unused duplicate RTL as design units in this independent reimport.
  xml=b/'ti60f225_oob.xml';s=xml.read_text()
  for n in ('rgb720_stream_writer.sv','axi_frame_write_dma.sv','rgb24_stream_pack.sv'):
   s,c=re.subn(r'\s*<efx:design_file name="'+re.escape(n)+r'" version="default" library="default"\s*/>', '',s);assert c==1
  xml.write_text(s,encoding='utf8')
  for row in m['inputs']:row['sha256']=sha(row['snapshot'])
  m.update(joint_parent_sha256=sha(PARENT/'inputs.json'),joint_transform_sha256=sha(TRANSFORM/'result.json'),joint_builder_sha256=sha(__file__),hardware_access=False)
  (b/'inputs.json').write_text(json.dumps(m,indent=2))
  for n in ('map.cmd','camera_graft.json','recovery_graft.json','capture_burst_graft.json'):shutil.copy2(PARENT/n,b/n)
 else:m=json.loads((b/'inputs.json').read_text())
 if a.prepare_only:print('PREPARED',b,flush=True);return
 for row in m['inputs']:assert sha(row['snapshot'])==row['sha256']
 if a.audit_only:body=(b/'map.log').read_text();seconds=None
 else:
  assert not (b/'map.log').exists()
  body,seconds=run(['cmd.exe','/d','/c','map.cmd'],b,'map.log',timeout=480)
 assert re.search(r'map\s*:\s*PASS',body)
 source=(b/'native_source.v').read_text();mapped=(b/'outflow/ti60f225_oob.map.v').read_text()
 sg,mg=graph(source),graph(mapped);missing=set(sg)-set(mg)
 # Only newly emitted unused combinational LUTs may disappear. Verify that
 # their outputs feed no retained cell; exclude them from the reimport proof.
 assert all(n.startswith('JOINT_DIV_') and sg[n][0]=='EFX_LUT4' for n in missing),sorted(missing)[:20]
 dead={t for n in missing for t in sg[n][1]['O']}
 assert not [(n,p,t) for n,(k,pins) in sg.items() if n not in missing for p,ts in pins.items() for t in ts if t in dead]
 clean=CELL.sub(lambda z:'' if z[2].lstrip('\\') in missing else z[0],source)
 # The mapper may omit an unconsumed terminal carry output. Never discard a
 # consumed port or general functional pin to make a connectivity check pass.
 omitted={}
 for n,(k,pins) in graph(clean).items():
  absent=set(pins)-set(mg[n][1])
  if k!='EFX_LUT4' and absent:
   assert n.startswith('JOINT_LOCAL_') and k=='EFX_ADD' and absent=={'CO'},(n,k,absent)
   tok=pins['CO'][0]
   assert sum(tok in ts for nn,(kk,pp) in sg.items() for pin,ts in pp.items())==1,(n,tok)
   omitted[n]=tok
 def omit_terminal(z):
  if z[2].lstrip('\\') not in omitted:return z[0]
  return re.sub(r',?\s*\.CO\([^()]*\)','',z[0])
 clean=CELL.sub(omit_terminal,clean)
 conn=verify(flattened_public_ports(clean),flattened_public_ports(mapped));assert conn['added_protected']==0
 lex=lexical_roundtrip(clean,mapped);boundary=compare_boundary(source,mapped,conn['renamed_nets'])
 (b/'public_connectivity_audit.json').write_text(json.dumps(conn,indent=2))
 old=resource_counts(PARENT/'outflow/ti60f225_oob.map.rpt');counts=resource_counts(b/'outflow/ti60f225_oob.map.rpt');expected=dict(old)
 for k,v in proof['added_kinds'].items():expected[k]=expected.get(k,0)+v
 for k,v in proof['removed_kinds'].items():expected[k]-=v
 expected['EFX_LUT4']-=len(missing)
 assert expected==counts,(expected,counts)
 assert protected_blocks(source)==protected_blocks((PARENT/'outflow/ti60f225_oob.map.v').read_text())
 assert counts.get('EFX_RAM10',0)+counts.get('EFX_DPRAM10',0)<=256
 assert counts.get('EFX_DSP48',0)+counts.get('EFX_DSP24',0)//2==160
 assert '[EFX-0654' not in (b/'outflow/ti60f225_oob.warn.log').read_text()
 for row in m['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 r=dict(status='CAMERA_JOINT_FULL_BOARD_MAP_PASS',seconds=seconds,counts=counts,lexical_audit=lex,protected_boundary=boundary,dead_generated_luts=sorted(missing),unconsumed_terminal_carry_omitted=omitted,
  added_cells=sorted(set(mg)-set(graph((PARENT/'outflow/ti60f225_oob.map.v').read_text()))),map_sha256=sha(b/'outflow/ti60f225_oob.map.v'),vdb_sha256=sha(b/'outflow/ti60f225_oob.vdb'),
  inputs_sha256=sha(b/'inputs.json'),evidence={str(f):sha(f) for f in (Path(__file__),TRANSFORM/'result.json',b/'public_connectivity_audit.json')},hardware_access=False,physical_pass=False,board_release=False)
 assert not (b/'result.json').exists();(b/'result.json').write_text(json.dumps(r,indent=2));print(r['status'],counts,flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  if BOARD.exists():(BOARD/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2))
  raise
