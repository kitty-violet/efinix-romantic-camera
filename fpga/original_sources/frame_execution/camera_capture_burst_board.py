"""Replace only the audited native camera writer; retain CPU/CNN/DDR graph.

The preserved graph is intentionally NOT regenerated from old assembly RTL.
Eight AWLEN selectors are extended; owner-locking and DDR bridge state stay.
"""
import argparse,collections,importlib.util,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from camera_recovery_board import BOARD as PARENT,audit_delta as parent_audit
from native_local_drivers import CELL,graph,raw_cells,render_token
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks,resource_counts
from map_native_local_drivers import flattened_public_ports
from audit_native_roundtrip import verify
from probe_native_open_ports import compare_boundary
BOARD=ROOT/'tmp/frame-execution/connected-board-camera-burst1'
VARIANT=HERE/'variants/camera_capture_burst/rgb720_stream_writer.sv'
P='video_processing_pipeline/pipeline/';C=P+'capture/';F=P+'fabric/shared_arbiter/'
RESET='u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/clk_rstn'
def net(n,i=None):return ('net',n,i)
def load_survey():
 f=ROOT/'tmp/board-test/survey-capture-graft.py'
 spec=importlib.util.spec_from_file_location('capture_survey',f);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);return s
def cut_plan(s):
 g,drivers,loads,OUT=s.g,s.drivers,s.loads,s.OUT
 provided={net(C+'active'):'cap_active',net(C+'dma/state',0):'cap_state[0]',net(C+'dma/state',1):'cap_state[1]',net('bringup_pipeline_status',1):'cap_error',net('n73194_2'):'cap_wvalid',net('n73203_2'):'cap_wlast',net('n17177'):"1'b0",net('n73210_2'):'cap_ready',net('n71233_2'):'cap_awvalid',net(C+'dma_done'):'cap_done',net(C+'dma_failed'):'cap_failed',net('n73854_2'):'!cap_failed',net('n73125_2'):"1'b0"}
 provided.update({net(P+'rgb_wdata',i):f'cap_wdata[{i}]' for i in range(128)})
 provided.update({net(P+'rgb_awaddr',i):f'cap_awaddr[{i}]' for i in range(4,32)})
 provided.update({net('s_axi_awlen',i):f'cap_selected ? cap_awlen[{i}]:cap_other_awlen[{i}]' for i in range(8)})
 removed={n for n,(k,p) in g.items() if k=='EFX_FF' and n.startswith(C)}|{drivers[t] for t in provided}
 while True:
  missing={t for n in removed for pin in OUT[g[n][0]] for t in g[n][1].get(pin,[]) if t not in provided}
  new={n for t in missing for n,pin in loads[t] if g[n][0] in ['EFX_LUT4','EFX_ADD']}
  if new<=removed:break
  removed|=new
 terminal=[(t,n,pin) for t in missing for n,pin in loads[t] if n not in removed]
 assert not terminal,terminal
 assert len(removed)==980 and not any(n.startswith('protected_inst/') for n in removed)
 assert all(not n.startswith(F) or g[n][0]!='EFX_FF' for n in removed)
 return provided,removed

def transform(s):
 provided,removed=cut_plan(s);old=s.text;raw=raw_cells(old);g=graph(old);ps=params(old)
 # Reuse the non-camera mux branches exactly. The former camera term is
 # forced to zero ONLY inside these copied muxes, before selecting new AWLEN.
 copies=[]
 for i in range(8):
  n=s.drivers[net('s_axi_awlen',i)];k,p,_=raw[n];assert k=='EFX_LUT4'
  p=dict(p);p['O']=f'cap_other_awlen[{i}]';new='CAP_OLD_OTHER_AWLEN_'+str(i)
  copies.append('    '+k+' '+new+' ('+', '.join('.'+pin+'('+v+')' for pin,v in p.items())+') /* unchanged non-camera mux */ ;')
  copies.extend(f"    defparam {new}.{key}={v[0]}'h{v[1]:x};" for key,v in ps[n].items())
 slots=json.loads((PARENT/'camera_graft.json').read_text())['diagnostic_slots'];prefix='cpu/system_hardJtag_debug_logic_jtagBridge/observation/'
 diag={prefix+f'held_payload[{832+slots[95+i]}]~FF':i for i in range(6)}
 def replace_cell(m):
  n=m[2].lstrip('\\')
  if n in removed:return ''
  if n in diag:
   t,count=re.subn(r'\.D\([^()]*\)',f'.D(cap_progress_page[{diag[n]}])',m[0]);assert count==1;return t
  return m[0]
 body=CELL.sub(replace_cell,old)
 body=re.sub(r'^\s*defparam\s+(\\\S+|\w+)\s*\.\w+\s*=[^;]+;[^\n]*',lambda m:'' if m[1].lstrip('\\') in removed else m[0],body,flags=re.M)
 assert set(params(body))==set(ps)-removed
 assigns=['    assign '+render_token(t)+' = '+expr+';' for t,expr in provided.items()]
 addition=r'''
    wire cap_active,cap_error,cap_ready,cap_done,cap_failed,cap_awvalid,cap_wvalid,cap_wlast;
    wire [1:0] cap_state;
    wire [127:0] cap_wdata;
    wire [31:0] cap_awaddr;
    wire [7:0] cap_awlen,cap_other_awlen;
    reg [5:0] cap_progress;
    wire [5:0] cap_progress_page;
    wire cap_reset = !RESET;
    wire cap_selected = SEL == 2'd0;
    wire cap_owned = OWNER == 2'd0;
    wire cap_awready = !cap_reset && !ACTIVE && SEL_VALID && cap_selected && s_axi_awready;
    wire cap_wready = !cap_reset && ACTIVE && !COMPLETE && cap_owned && s_axi_wready;
    wire cap_bvalid = !cap_reset && ACTIVE && cap_owned && s_axi_bvalid;
    wire cap_free0 = SLOTS[2:0]==0;
    wire cap_free1 = SLOTS[5:3]==0;
    wire cap_free2 = SLOTS[8:6]==0;
    wire cap_frame_ready = !cap_reset && !bringup_pipeline_status[0] && !CAPTURING && (cap_free0||cap_free1||cap_free2);
    wire [31:0] cap_frame_address = cap_free0 ? 32'h0 : cap_free1 ? 32'h400000 : cap_free2 ? 32'h800000 : 32'h0;
    rgb720_stream_writer #(.MAX_CAPTURE_BURST(16)) camera_burst_capture (
      .clk(process_clk_150),.reset(cap_reset),.cancel(control_cancel || STOPPED),.frame_abort(raw_capture_abort),
      .s_valid(DOWN_VALID),.s_ready(cap_ready),.s_rgb(DOWN_RGB ^ 24'h808080),.s_frame_id(DOWN_ID),
      .s_sof(DOWN_SOF),.s_eol(DOWN_EOL),.s_eof(DOWN_EOF),
      .frame_request(),.frame_ready(cap_frame_ready),.frame_address(cap_frame_address),.requested_frame_id(),
      .frame_done(cap_done),.frame_failed(cap_failed),.busy(),.error(cap_error),
      .awaddr(cap_awaddr),.awlen(cap_awlen),.awvalid(cap_awvalid),.awready(cap_awready),
      .wdata(cap_wdata),.wstrb(),.wlast(cap_wlast),.wvalid(cap_wvalid),.wready(cap_wready),
      .bresp(2'b00),.bvalid(cap_bvalid),.bready(),.graft_active(cap_active),.graft_dma_state(cap_state));
    always @(posedge process_clk_150) begin
      if(cap_reset) cap_progress<=0;
      else cap_progress<=cap_progress | {cap_failed,cap_done,cap_wvalid&&cap_wready,cap_awvalid&&cap_awready,CAP_START,DOWN_VALID};
    end
'''
 replacements={'RESET':RESET,'SEL':F+'write_select_owner','OWNER':F+'write_owner','ACTIVE':F+'write_active','SEL_VALID':F+'write_select_valid','COMPLETE':F+'write_data_complete','SLOTS':'video_processing_pipeline/frame_states','CAPTURING':P+'owner/capturing','STOPPED':P+'stopped','DOWN_VALID':'video_processing_pipeline/down_valid','DOWN_RGB':'video_processing_pipeline/down_rgb','DOWN_ID':'video_processing_pipeline/down_frame_id','DOWN_SOF':'video_processing_pipeline/downscaler/out_sof','DOWN_EOL':'video_processing_pipeline/downscaler/out_eol','DOWN_EOF':'video_processing_pipeline/downscaler/out_eof','CAP_START':C+'start_fire'}
 addition=re.sub(r'\b('+ '|'.join(replacements)+r')\b',lambda m:render_token(net(replacements[m[0]])),addition)
 for n,i in diag.items():addition+=f'\n    assign cap_progress_page[{i}] = '+render_token(net(prefix+'request_sync'))+f' ? cap_progress[{i}] : '+raw[n][1]['D']+';\n'
 anchor=body.index('    protected_cell protected_inst (');body=body[:anchor]+'\n'.join(copies+assigns)+addition+body[anchor:]
 after=graph(body);pp=params(body)
 assert set(g)-set(after)==removed
 for n in set(g)-removed:
  assert ps[n]==pp[n],n
  if n in diag:assert {p:t for p,t in g[n][1].items() if p!='D'}=={p:t for p,t in after[n][1].items() if p!='D'},n
  else:assert g[n]==after[n],n
 assert protected_blocks(old)==protected_blocks(body)
 return body,dict(removed=sorted(removed),provided=[dict(token=t,expression=e) for t,e in provided.items()],copies=[f'CAP_OLD_OTHER_AWLEN_{i}' for i in range(8)],changed_snapshot_D=diag,progress_bits={'95':'down_valid_seen','96':'capture_started','97':'AW_accepted','98':'W_accepted','99':'frame_done','100':'frame_failed'},protected_blocks_unchanged=True,unchanged_retained_cells=len(g)-len(removed)-len(diag))

def expanded_sources():
 writer=VARIANT.read_text();dma=(PARENT/'assembly/axi_frame_write_dma.sv').read_text()
 # Expose existing state for the preserved native busy/owner glue. These
 # output ports introduce no new sequential behavior or alternate controls.
 writer=writer.replace('input wire [1:0] bresp,input wire bvalid,output wire bready','input wire [1:0] bresp,input wire bvalid,output wire bready,\n output wire graft_active,output wire [1:0] graft_dma_state')
 writer=writer.replace('.config_error(dma_config_error),.protocol_error(dma_protocol_error));','.config_error(dma_config_error),.protocol_error(dma_protocol_error),.graft_state(graft_dma_state));')
 writer=writer.replace('endmodule',' assign graft_active=active;\nendmodule')
 dma=dma.replace('output reg                          protocol_error','output reg                          protocol_error,\n    output wire [1:0]                   graft_state')
 dma=dma.replace('endmodule',' assign graft_state=state;\nendmodule')
 assert writer.count('graft_active')==2 and writer.count('graft_dma_state')==2 and dma.count('graft_state')==2
 return {'rgb720_stream_writer.sv':writer,'axi_frame_write_dma.sv':dma,'rgb24_stream_pack.sv':(PARENT/'assembly/rgb24_stream_pack.sv').read_text()}

def audit_delta(board):
 parent_audit(PARENT);m=json.loads((board/'inputs.json').read_text());r=json.loads((board/'result.json').read_text())
 assert r['status']=='CAMERA_BURST_FULL_BOARD_MAP_PASS'
 assert m['burst_parent_sha256']==sha(PARENT/'inputs.json') and m['burst_builder_sha256']==sha(__file__)
 for row in m['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 for f,h in r['evidence'].items():assert sha(f)==h,f
 assert r['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v') and r['vdb_sha256']==sha(board/'outflow/ti60f225_oob.vdb')
 return dict(status='CAMERA_BURST_FULL_VIDEO_A_MAP_BOUND',map_sha256=r['map_sha256'],hardware_access=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=BOARD);p.add_argument('--prepare-only',action='store_true');p.add_argument('--map-existing',action='store_true');p.add_argument('--cut-proof',type=Path,required=True);a=p.parse_args();board=a.root.resolve();assert board.is_relative_to(ROOT/'tmp')
 parent_audit(PARENT);proof=json.loads(a.cut_proof.read_text());assert proof['status']=='PASS' and proof['map_sha256']==sha(PARENT/'outflow/ti60f225_oob.map.v')
 for f,h in proof['inputs'].items():assert sha(f)==h
 s=load_survey();body,graft=transform(s)
 if not a.map_existing:
  assert not board.exists();board.mkdir();m=json.loads((PARENT/'inputs.json').read_text())
  for row in m['inputs']:
   src=Path(row['snapshot']);dst=board/src.relative_to(PARENT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);row['snapshot']=str(dst)
  (board/'native_source.v').write_text(body,encoding='utf8')
  for n,t in expanded_sources().items():
   (board/n).write_text(t,encoding='utf8');m['inputs'].append(dict(source=str(VARIANT if n=='rgb720_stream_writer.sv' else PARENT/'assembly'/n),snapshot=str(board/n),sha256=sha(board/n)))
  xml=board/'ti60f225_oob.xml';x=xml.read_text();anchor='<efx:design_file name="native_source.v" version="default" library="default" />';assert x.count(anchor)==1
  x=x.replace(anchor,anchor+'\n'+''.join(f'<efx:design_file name="{n}" version="default" library="default"/>\n' for n in expanded_sources()));xml.write_text(x)
  for row in m['inputs']:row['sha256']=sha(row['snapshot'])
  m.update(burst_parent=str(PARENT),burst_parent_sha256=sha(PARENT/'inputs.json'),burst_builder_sha256=sha(__file__),burst_variant_sha256=sha(VARIANT),burst_cut_proof=str(a.cut_proof.resolve()),burst_cut_proof_sha256=sha(a.cut_proof))
  (board/'inputs.json').write_text(json.dumps(m,indent=2));(board/'capture_burst_graft.json').write_text(json.dumps(graft,indent=2))
  for n in ['map.cmd','camera_graft.json','recovery_graft.json']:shutil.copy2(PARENT/n,board/n)
 else:
  assert (board/'native_source.v').read_text()==body and not (board/'map.log').exists();m=json.loads((board/'inputs.json').read_text());assert m['burst_builder_sha256']==sha(__file__)
 if a.prepare_only:print('PREPARED_NO_MAP',board,len(graft['removed']));return
 for row in m['inputs']:assert sha(row['snapshot'])==row['sha256']
 log,secs=run(['cmd.exe','/d','/c','map.cmd'],board,'map.log',timeout=480);assert re.search(r'map\s*:\s*PASS',log)
 mapped=(board/'outflow/ti60f225_oob.map.v').read_text();ng=graph(mapped);after=graph(body);added=set(ng)-set(after);missing=set(after)-set(ng)
 assert not missing,sorted(missing)
 filtered=CELL.sub(lambda m:m[0] if m[2].lstrip('\\') in after else '',mapped)
 conn=verify(flattened_public_ports(body),flattened_public_ports(filtered));boundary=compare_boundary(body,mapped,conn['renamed_nets'])
 (board/'public_connectivity_audit.json').write_text(json.dumps(conn,indent=2))
 counts=resource_counts(board/'outflow/ti60f225_oob.map.rpt');assert counts['EFX_RAM10']+counts['EFX_DPRAM10']<=256 and counts['EFX_DSP48']+counts['EFX_DSP24']//2==160
 evidence={str(f.resolve()):sha(f) for f in [a.cut_proof,Path(__file__),board/'capture_burst_graft.json',board/'public_connectivity_audit.json',VARIANT]}
 result=dict(status='CAMERA_BURST_FULL_BOARD_MAP_PASS',seconds=secs,counts=counts,removed=graft['removed'],added_cells=sorted(added),inputs_sha256=sha(board/'inputs.json'),map_sha256=sha(board/'outflow/ti60f225_oob.map.v'),vdb_sha256=sha(board/'outflow/ti60f225_oob.vdb'),protected_boundary=boundary,evidence=evidence,physical_pass=False,board_release=False,hardware_access=False)
 (board/'result.json').write_text(json.dumps(result,indent=2));(board/'delta_audit.json').write_text(json.dumps(audit_delta(board),indent=2));print(result['status'],counts,secs)
if __name__=='__main__':main()
