"""Camera-only native graft, preserving the proved full video/A public graph."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from local_driver_board import BASE,audit_delta as parent_audit
from native_local_drivers import CELL,graph
from map_native_local_drivers import flattened_public_ports,lexical_roundtrip
from audit_native_roundtrip import verify
from probe_native_open_ports import compare_boundary
from complete_native_logic_split import protected_blocks,resource_counts
PARENT=ROOT/'tmp/frame-execution/connected-board-local-drivers1'
BOARD=ROOT/'tmp/frame-execution/connected-board-camera-checked2'
UNIT=ROOT/'tmp/board-test/camera-checked-unit2'
ADDED=('camera_checked_config.sv','camera_input_observer.sv')

def graft(text):
 g=graph(text);edits={}
 for n,net in [('LUT__168106','camera_new_sda_oen'),('LUT__168107','camera_new_scl_oen')]:
  assert g[n][0]=='EFX_LUT4';assert list(g[n][1])==['I0','O']
  edits[n]=('I0',net)
 prefix='cpu/system_hardJtag_debug_logic_jtagBridge/observation/'
 slots=[i for i in range(128) if prefix+f'held_payload[{832+i}]~FF' in g]
 assert len(slots)==111
 # The old observer optimized constants/aliases out of 17 bits. Pack the
 # lower 111 diagnostic bits only into actual FFs; host decodes this map.
 for i,slot in enumerate(slots):
  n=prefix+f'held_payload[{832+slot}]~FF'
  assert g[n][0]=='EFX_FF' and g[n][1]['CLK']==[('net','process_clk_150',None)]
  edits[n]=('D',f'camera_diagnostic[{i}]')
 def change(m):
  name=m[2].lstrip('\\')
  if name not in edits:return m[0]
  p,v=edits[name];body,n=re.subn(r'\.'+p+r'\([^()]*\)',lambda _:'.'+p+'('+v+')',m[0]);assert n==1
  return body
 changed=CELL.sub(change,text)
 # The original valid signal is DE & VC0-HS; recover the HS cofactor from
 # these two exact LUT truth tables, without changing their existing loads.
 assert g['LUT__163046'][1]['I2']==[('net','w_mipi_rx_de',None)]
 assert g['LUT__163047'][1]['I2']==g['LUT__163046'][1]['O']
 from probe_full_native_split_map import params
 ps=params(text);assert ps['LUT__163046']['LUTMASK']==(16,0x1010) and ps['LUT__163047']['LUTMASK']==(16,0x7070)
 for value in range(32):
  a,b,de,c,d=[value>>i&1 for i in range(5)]
  first=0x1010>>(a+2*b+4*de)&1;valid=0x7070>>(c+2*d+4*first)&1
  assert valid==int(de and not a and not b and not(c and d))
 addition=r'''
    wire camera_new_sda_oen,camera_new_scl_oen,camera_config_ok,camera_config_failed;
    wire [63:0] camera_config_diagnostic;
    wire [127:0] camera_diagnostic;
    wire camera_observed_hs = !(\inst_efx_csi2_rx/u_efx_csi2_rx/csi2_rx_top_inst/genblk3.byte2pixel_inst/vc_sync [1] |
                                \inst_efx_csi2_rx/u_efx_csi2_rx/csi2_rx_top_inst/genblk3.byte2pixel_inst/vc_sync [0]) && !(n75722_2 && n75725_2);
    camera_checked_config camera_checked(
      .clk(CLK_5M),.reset_n(w_i2c_arstn),.scl_i(io_cam_scl_IN),.sda_i(io_cam_sda_IN),
      .scl_o(),.scl_oen(camera_new_scl_oen),.sda_o(),.sda_oen(camera_new_sda_oen),
      .config_ok(camera_config_ok),.config_failed(camera_config_failed),.diagnostic(camera_config_diagnostic));
    camera_input_observer camera_observer(
      .system_clk(process_clk_150),.system_reset_n(w_arstn),.pixel_clk(i_mipi_rx_pclk),.pixel_reset_n(w_mipi_rx_pclk_arstn),
      .config_diagnostic(camera_config_diagnostic),.config_ok(camera_config_ok),.config_failed(camera_config_failed),
      .vs(w_mipi_rx_vs),.hs(camera_observed_hs),.de(w_mipi_rx_de),
      .raw_errors(raw_stream_errors),.boundary_error(camera_boundary_error),.diagnostic(camera_diagnostic));
'''
 anchor=changed.index('    protected_cell protected_inst (');changed=changed[:anchor]+addition+changed[anchor:]
 after=graph(changed);assert g.keys()==after.keys()
 for n in g:
  a,b=g[n],after[n];assert a[0]==b[0]
  for p in a[1]:
   if n in edits and p==edits[n][0]:continue
   assert a[1][p]==b[1][p],(n,p)
 assert params(changed)==ps and protected_blocks(changed)==protected_blocks(text)
 return changed,dict(changed_pins={n:p for n,(p,v) in edits.items()},diagnostic_slots=slots,diagnostic_bits=111,
   unchanged_cells=len(g)-len(edits),changed_cells=len(edits),protected_bytes_unchanged=True,
   hs_cofactor_truth_checks=32,sole_pad_owner='camera_checked',legacy_i2c_disconnected_from_pad_enable=True)

def audit_delta(board):
 parent_audit(PARENT)
 m=json.loads((board/'inputs.json').read_text());r=json.loads((board/'result.json').read_text())
 assert m['camera_parent']==str(PARENT) and m['camera_parent_sha256']==sha(PARENT/'inputs.json')
 assert m['camera_builder_sha256']==sha(__file__)
 for row in m['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 for f,h in r['evidence'].items():assert sha(f)==h,f
 assert r['status']=='CAMERA_CHECKED_FULL_BOARD_MAP_PASS' and r['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
 assert r['vdb_sha256']==sha(board/'outflow/ti60f225_oob.vdb')
 return dict(status='CAMERA_ONLY_CHANGE_FULL_VIDEO_A_MAP_BOUND',map_sha256=r['map_sha256'],
   graph_sha256=sha(board/'public_connectivity_audit.json'),graft_sha256=sha(board/'camera_graft.json'),hardware_access=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=BOARD);a=p.parse_args();board=a.root.resolve()
 assert board.is_relative_to(ROOT/'tmp') and not board.exists();parent_audit(PARENT)
 u=json.loads((UNIT/'result.json').read_text());assert u['status']=='CHECKED_CAMERA_PIN_TESTS_PASS'
 for f,h in u['inputs'].items():assert sha(f)==h
 board.mkdir();meta=json.loads((PARENT/'inputs.json').read_text())
 try:
  for row in meta['inputs']:
   old=Path(row['snapshot']);dest=board/old.relative_to(PARENT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,dest)
   row['snapshot']=str(dest)
  source=board/'native_source.v';body,proof=graft(source.read_text());source.write_text(body,encoding='utf-8')
  for name in ADDED:shutil.copy2(HERE/name,board/name)
  xml=board/'ti60f225_oob.xml';s=xml.read_text();anchor='<efx:design_file name="native_source.v" version="default" library="default" />'
  names=list(ADDED)+['rtl/cam_i2c_ctrl/i2c/i2c_master_reg_rom.v','rtl/cam_i2c_ctrl/i2c_master/i2c_master_byte_ctrl.v','rtl/cam_i2c_ctrl/i2c_master/i2c_master_bit_ctrl.v']
  assert s.count(anchor)==1;s=s.replace(anchor,anchor+'\n'+''.join(f'<efx:design_file name="{n}" version="default" library="default" />\n' for n in names));xml.write_text(s,encoding='utf-8')
  # Include files are local to the existing byte/bit sources; also provide top-level includes.
  for n in ('i2c_master_defines.v','timescale.v'):shutil.copy2(board/'rtl/cam_i2c_ctrl/i2c_master'/n,board/n)
  for row in meta['inputs']:row['sha256']=sha(row['snapshot'])
  for n in (*ADDED,'i2c_master_defines.v','timescale.v'):
   meta['inputs'].append(dict(source=str(board/n),snapshot=str(board/n),sha256=sha(board/n)))
  meta.update(camera_parent=str(PARENT),camera_parent_sha256=sha(PARENT/'inputs.json'),camera_builder_sha256=sha(__file__),hardware_access=False)
  (board/'inputs.json').write_text(json.dumps(meta,indent=2));(board/'camera_graft.json').write_text(json.dumps(proof,indent=2))
  shutil.copy2(ROOT/'tmp/frame-execution/native-local-drivers-map2/map.cmd',board/'map.cmd')
  log,seconds=run(['cmd.exe','/d','/c','map.cmd'],board,'map.log',timeout=480);assert re.search(r'map\s*:\s*PASS',log)
  mapped=(board/'outflow/ti60f225_oob.map.v').read_text();oldcells=set(graph(body));newcells=set(graph(mapped))-oldcells
  assert newcells and all(n.startswith(('camera_checked/','camera_observer/')) or re.fullmatch(r'(?:LUT__\d+|ADD__\d+|SUB__\d+)',n) for n in newcells),sorted(newcells)[:30]
  filtered=CELL.sub(lambda m:m[0] if m[2].lstrip('\\') in oldcells else '',mapped)
  comparison=verify(flattened_public_ports(body),flattened_public_ports(filtered))
  lexical=lexical_roundtrip(body,filtered)
  boundary=compare_boundary(body,mapped,comparison['renamed_nets'])
  (board/'public_connectivity_audit.json').write_text(json.dumps(comparison,indent=2))
  result=dict(status='CAMERA_CHECKED_FULL_BOARD_MAP_PASS',map_sha256=sha(board/'outflow/ti60f225_oob.map.v'),
   vdb_sha256=sha(board/'outflow/ti60f225_oob.vdb'),resources=resource_counts(board/'outflow/ti60f225_oob.map.rpt'),
   seconds=seconds,new_cells=sorted(newcells),lexical_audit=lexical,protected_boundary=boundary,
   evidence={str(f):sha(f) for f in (UNIT/'result.json',board/'inputs.json',board/'camera_graft.json',board/'public_connectivity_audit.json',HERE/'camera_checked_board.py')},
   board_download_allowed=False,hardware_access=False)
  (board/'result.json').write_text(json.dumps(result,indent=2));(board/'delta_audit.json').write_text(json.dumps(audit_delta(board),indent=2))
  print(result['status'],result['resources'],flush=True)
 except Exception as e:
  (board/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
