"""Replace the proved 15-cell camera boundary cone; preserve full A/DDR graph."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from camera_ready_board import BOARD as PARENT,audit_delta as parent_audit
from native_local_drivers import CELL,graph,raw_cells,render_token
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks,resource_counts
from map_native_local_drivers import flattened_public_ports
from audit_native_roundtrip import verify
from probe_native_open_ports import compare_boundary
BOARD=ROOT/'tmp/frame-execution/connected-board-camera-recovery1'
VARIANT=HERE/'variants/camera_boundary_recovery'
def audit_delta(board):
 parent_audit(PARENT);m=json.loads((board/'inputs.json').read_text());r=json.loads((board/'result.json').read_text())
 assert m['recovery_parent']==str(PARENT) and m['recovery_parent_sha256']==sha(PARENT/'inputs.json')
 assert m['recovery_builder_sha256']==sha(ROOT/'tmp/board-test/camera_recovery_board_prepared.py')
 assert r['auditor_sha256']==sha(__file__)
 for row in m['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 for f,h in r['evidence'].items():assert sha(f)==h,f
 assert r['status']=='CAMERA_RECOVERY_FULL_BOARD_MAP_PASS' and r['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v') and r['vdb_sha256']==sha(board/'outflow/ti60f225_oob.vdb')
 return dict(status='CAMERA_RECOVERY_FULL_VIDEO_A_MAP_BOUND',map_sha256=r['map_sha256'],hardware_access=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=BOARD);p.add_argument('--prepare-only',action='store_true');p.add_argument('--map-existing',action='store_true');a=p.parse_args()
 board=a.root.resolve();assert board.is_relative_to(ROOT/'tmp');parent_audit(PARENT)
 checks=[ROOT/'tmp/frame-execution/camera-boundary-recovery1/result.json',ROOT/'tmp/frame-execution/camera-fault-observer1/result.json']
 for f in checks:
  r=json.loads(f.read_text());assert r['status']=='PASS'
  for n,h in r['inputs'].items():assert sha(n)==h,n
 old=(PARENT/'outflow/ti60f225_oob.map.v').read_text();g=graph(old);ps=params(old)
 survey=ROOT/'tmp/board-test/camera-recovery-graft-survey1.json';s=json.loads(survey.read_text());assert s['map_sha256']==sha(PARENT/'outflow/ti60f225_oob.map.v')
 removed=set(s['removed']);assert len(removed)==15 and not s['extra_loads']
 prefix='cpu/system_hardJtag_debug_logic_jtagBridge/observation/'
 slots=json.loads((PARENT/'camera_graft.json').read_text())['diagnostic_slots'];assert len(slots)==111
 snap_names=[prefix+f'held_payload[{832+i}]~FF' for i in slots]
 raw=raw_cells(old);old_page=[raw[n][1]['D'] for n in snap_names]
 # Exact source release used by the existing RAW packer, including polarity.
 assert g['raw_frontend/ingress/packer/state[0]~FF'][1]['SR']==[('net','raw_frontend/ingress/n10',None)]
 assert ps['raw_frontend/ingress/packer/state[0]~FF']['SR_POLARITY']==(1,0)
 assert g['LUT__163505'][1]==dict(**{f'I{i}':[('net','raw_frontend/ingress/camera_release',i)] for i in range(4)},O=[('net','raw_frontend/ingress/n10',None)]) and ps['LUT__163505']['LUTMASK']==(16,0x8000)
 hs=raw['camera_observer/hs_d~FF'][1]['D']
 assert g['camera_polarity_known~FF'][1]['CE']==[('net','n1345',None)]
 edits={n:i for i,n in enumerate(snap_names)}
 def change(m):
  n=m[2].lstrip('\\')
  if n in removed:return ''
  if n not in edits:return m[0]
  body,k=re.subn(r'\.D\([^()]*\)',f'.D(camera_stream_page[{edits[n]}])',m[0]);assert k==1;return body
 body=CELL.sub(change,old)
 for n in removed:
  body=re.sub(r'^\s*defparam\s+'+re.escape('\\'+n if '/' in n or '~' in n else n)+r'\s*\.[^;]+;[^\n]*','',body,flags=re.M)
 assert not (set(params(body)) & removed)
 addition=r'''
    wire camera_recovery_reset_n = \raw_frontend/ingress/n10 ;
    wire [110:0] camera_stream_page,camera_stream_faults;
    wire [110:0] camera_original_page = {OLD_PAGE};
    assign camera_stream_page = \cpu/system_hardJtag_debug_logic_jtagBridge/observation/request_sync ? camera_stream_faults : camera_original_page;
    video_frame_boundary_detector camera_recovery_boundary(
      .clk(i_mipi_rx_pclk),.rst_n(camera_recovery_reset_n),.vsync(w_mipi_rx_vs),.data_enable(n1345),
      .frame_start(camera_frame_start),.frame_end(camera_frame_end),.pixel_valid(camera_pixel_valid),
      .frame_active(camera_frame_active),.polarity_known(camera_polarity_known),
      .active_video_vsync_level(camera_active_vsync_level),.boundary_error(camera_boundary_error));
    camera_stream_fault_observer camera_stream_observer(
      .system_clk(process_clk_150),.system_reset_n(w_arstn),.pixel_clk(i_mipi_rx_pclk),.pixel_reset_n(camera_recovery_reset_n),
      .vs(w_mipi_rx_vs),.hs(HS_TAP),.de(w_mipi_rx_de),.known(camera_polarity_known),.level(camera_active_vsync_level),
      .frame_active(camera_frame_active),.fs(camera_frame_start),.fe(camera_frame_end),.pv(camera_pixel_valid),
      .packer_ready(\raw_frontend/ingress/crossing/wr_guard_done && !\raw_frontend/ingress/crossing/fifo_full ),
      .boundary_error(camera_boundary_error),.length_error(raw_stream_errors[7]),
      .packer_words({1'b0,\raw_frontend/ingress/packer/captured_words [18:0]}),
      .capture_active(\video_processing_pipeline/pipeline/capture/active ),
      .owner_states(\video_processing_pipeline/frame_states ),.diagnostic(camera_stream_faults));
'''.replace('OLD_PAGE',', '.join(reversed(old_page))).replace('HS_TAP',hs)
 anchor=body.index('    protected_cell protected_inst (');body=body[:anchor]+addition+body[anchor:]
 after=graph(body);new_params=params(body);assert set(g)-set(after)==removed and not(set(after)-set(g))
 for n in after:
  assert ps[n]==new_params[n] if n in ps else True
  if n in edits:
   assert {p:t for p,t in g[n][1].items() if p!='D'}=={p:t for p,t in after[n][1].items() if p!='D'}
  else:assert g[n]==after[n],n
 assert protected_blocks(body)==protected_blocks(old)
 # Every added tap must be an actual driven bit; no source-level aliases.
 outputs={'EFX_FF':['Q'],'EFX_LUT4':['O'],'EFX_ADD':['O','CO'],'EFX_RAM10':['RDATA']}
 driven={t for k,p in g.values() for pin in outputs.get(k,[]) for t in p.get(pin,[])}
 taps=[('net','raw_frontend/ingress/packer/captured_words',i) for i in range(19)]+[('net','video_processing_pipeline/frame_states',i) for i in range(9)]
 taps += [('net',n,None) for n in ['raw_frontend/ingress/n10','raw_frontend/ingress/crossing/wr_guard_done','raw_frontend/ingress/crossing/fifo_full','video_processing_pipeline/pipeline/capture/active',prefix+'request_sync']]+[('net','raw_stream_errors',7)]
 assert all(t in driven for t in taps),[t for t in taps if t not in driven]
 if a.map_existing:
  # Removing adjacent defparam blocks in set order may leave a different
  # number of empty lines. Permit ONLY empty-line/CRLF differences here;
  # every nonempty line (including escaped identifiers) remains exact.
  prepared=(board/'native_source.v').read_text()
  if prepared!=body:
   import difflib
   difference=list(difflib.unified_diff(prepared.splitlines(),body.splitlines(),n=1))
   (board/'prepare_text_comparison.json').write_text(json.dumps(dict(equal_nonempty_lines=[x for x in prepared.splitlines() if x.strip()]==[x for x in body.splitlines() if x.strip()],difference=difference[:120]),indent=2))
  assert [x for x in prepared.splitlines() if x.strip()]==[x for x in body.splitlines() if x.strip()]
  body=prepared
  m=json.loads((board/'inputs.json').read_text());assert not (board/'map.log').exists()
 else:
  assert not board.exists();board.mkdir();m=json.loads((PARENT/'inputs.json').read_text())
  for row in m['inputs']:
   src=Path(row['snapshot']);dst=board/src.relative_to(PARENT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);row['snapshot']=str(dst)
  (board/'native_source.v').write_text(body,encoding='utf8')
  for n in ['video_frame_boundary_detector.sv','camera_stream_fault_observer.sv']:
   shutil.copy2(VARIANT/n,board/n);m['inputs'].append(dict(source=str(VARIANT/n),snapshot=str(board/n),sha256=sha(board/n)))
  xml=board/'ti60f225_oob.xml';x=xml.read_text();anchor='<efx:design_file name="native_source.v" version="default" library="default" />';assert x.count(anchor)==1
  x=x.replace(anchor,anchor+'\n'+''.join(f'<efx:design_file name="{n}" version="default" library="default" />\n' for n in ['video_frame_boundary_detector.sv','camera_stream_fault_observer.sv']));xml.write_text(x)
  for row in m['inputs']:row['sha256']=sha(row['snapshot'])
  m.update(recovery_parent=str(PARENT),recovery_parent_sha256=sha(PARENT/'inputs.json'),recovery_builder_sha256=sha(__file__))
  (board/'inputs.json').write_text(json.dumps(m,indent=2));shutil.copy2(PARENT/'map.cmd',board/'map.cmd');shutil.copy2(PARENT/'camera_graft.json',board/'camera_graft.json')
  (board/'recovery_graft.json').write_text(json.dumps(dict(removed=sorted(removed),changed_snapshot_D=snap_names,taps=taps,source_sha256=sha(board/'native_source.v'),parent_map_sha256=sha(PARENT/'outflow/ti60f225_oob.map.v'),page='E phase reserved; original config page otherwise unchanged',protected_blocks_unchanged=True),indent=2))
 if a.prepare_only:print('PREPARED_NO_MAP',board);return
 for row in m['inputs']:assert sha(row['snapshot'])==row['sha256']
 log,secs=run(['cmd.exe','/d','/c','map.cmd'],board,'map.log',timeout=480);assert re.search(r'map\s*:\s*PASS',log)
 mapped=(board/'outflow/ti60f225_oob.map.v').read_text();ng=graph(mapped);added=set(ng)-set(after);missing=set(after)-set(ng)
 assert not missing,sorted(missing)
 filtered=CELL.sub(lambda m:m[0] if m[2].lstrip('\\') in after else '',mapped)
 conn=verify(flattened_public_ports(body),flattened_public_ports(filtered));boundary=compare_boundary(body,mapped,conn['renamed_nets'])
 (board/'public_connectivity_audit.json').write_text(json.dumps(conn,indent=2))
 counts=resource_counts(board/'outflow/ti60f225_oob.map.rpt');assert counts['EFX_RAM10']+counts['EFX_DPRAM10']==228 and counts['EFX_DSP48']+counts['EFX_DSP24']//2==160
 evidence={str(f):sha(f) for f in [*checks,survey,Path(__file__),board/'recovery_graft.json',board/'public_connectivity_audit.json']}
 result=dict(status='CAMERA_RECOVERY_FULL_BOARD_MAP_PASS',auditor_sha256=sha(__file__),seconds=secs,counts=counts,removed=sorted(removed),added_cells=sorted(added),inputs_sha256=sha(board/'inputs.json'),map_sha256=sha(board/'outflow/ti60f225_oob.map.v'),vdb_sha256=sha(board/'outflow/ti60f225_oob.vdb'),protected_boundary=boundary,evidence=evidence,physical_pass=False,board_release=False,hardware_access=False)
 (board/'result.json').write_text(json.dumps(result,indent=2));(board/'delta_audit.json').write_text(json.dumps(audit_delta(board),indent=2));print(result['status'],counts,secs)
if __name__=='__main__':main()
