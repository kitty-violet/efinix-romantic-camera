"""Independent 68 camera-input candidate; preserves the frozen 67 release."""
import argparse,json,re,shutil,pickle
from pathlib import Path
from run_units import ROOT,HERE,sha
from native_local_drivers import CELL,graph,rewrite_expression
from continuous_control_native import connections,net
from audit_local_reset_implementation import norm
from complete_native_logic_split import protected_blocks
from validate_video_input_lut import tables
from video_service_display_native import split_net_declarations

BASE=ROOT/'tmp/frame-execution/video-service-native67-color-exposure'
BOARD=ROOT/'tmp/frame-execution/connected-board-video-service-probe58-color-exposure'
OUT=ROOT/'tmp/frame-execution/video-service-native68-input-isp'
MAP=ROOT/'tmp/frame-execution/connected-board-video-service-probe59-input-isp'
S='raw_frontend/consumer/serializer/'

def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def once(s,a,b):assert s.count(a)==1,(a[:80],s.count(a));return s.replace(a,b)

def main():
 global OUT,MAP
 parser=argparse.ArgumentParser();parser.add_argument('--native',type=Path);parser.add_argument('--board',type=Path);args=parser.parse_args()
 if args.native:OUT=args.native.resolve()
 if args.board:MAP=args.board.resolve()
 assert OUT.is_relative_to(ROOT/'tmp/frame-execution') and MAP.is_relative_to(ROOT/'tmp/frame-execution')
 assert not MAP.exists()
 if OUT.exists():assert sorted(p.name for p in OUT.iterdir())==['isp_b.hex','isp_g.hex','isp_r.hex']
 assert sha(BASE/'native_source.v')=='d0dee74938993379a99514b7098f9ef521f184e6cbeeb662b4c7d22aef5147a9'
 for test in ['video-input-color3','video-input-lut1','video-input-recovery-wide3']:
  assert json.loads((ROOT/'tmp/frame-execution'/test/'result.json').read_text())['status']=='PASS'
 original=(BASE/'native_source.v').read_text(encoding='utf8')
 h,g,params=pickle.load(open(BASE/'color-source-parse.pkl','rb'));assert h==sha(BASE/'native_source.v')
 drivers,loads=connections(g);edits={}
 def edit(n,p,old,new):
  assert old in g[n][1][p],(n,p,old);assert old not in edits.get(n,{}).get(p,{})
  edits.setdefault(n,{}).setdefault(p,{})[old]=new
 # Physical p3,p2,p1,p0 packing: swap each RAW pair before the legacy gain.
 for bit in range(16):
  old=net('raw_frontend/consumer/rd',bit)
  for n,p in loads[old]:
   assert n.startswith('raw_frontend/consumer/debayer/legacy_debayer/inst_rgb_gain/'),n
   edit(n,p,old,net(old[1],bit^8))
 # RGB/id move together through one elastic RAM lookup stage.
 for name,width,new in [('pair_pixel_rgb',24,'service_isp_rgb'),('pair_pixel_frame_id',16,'service_isp_id')]:
  for bit in range(width):
   old=net(name,bit)
   for n,p in loads[old]:
    assert n.startswith('video_processing_pipeline/downscaler/'),n
    edit(n,p,old,net(new,bit))
 # Original synthesis shared the serializer handshake and downscaler CE.
 # Split it explicitly; leave every serializer state/FIFO CE on input fire.
 old=net('~ceg_net7945')
 retained=[]
 for n,p in loads[old]:
  if n==S+'second_pending~FF':retained.append((n,p));continue
  assert n.startswith('video_processing_pipeline/downscaler/') or n in ['LUT__153494','LUT__153498','LUT__153504','LUT__153506'],n
  edit(n,p,old,net('service_isp_fire'))
 assert retained==[(S+'second_pending~FF','CE')]
 for n in ['LUT__153493','LUT__153496']:edit(n,'I3',net('n74389_2'),net('service_isp_ready'))
 assert params['LUT__153497']['LUTMASK']==(16,0x8888)
 edit('LUT__153497','I0',net(S+'fifo_data',66),net('service_isp_sof'))
 edit('LUT__153497','I1',net(S+'pair_fifo/dequeue'),net('service_isp_fire'))
 def change(m):
  n=norm(m[2])
  if n not in edits:return m[0]
  body=re.sub(r'\.(\w+)\(([^()]*)\)',lambda q:'.'+q[1]+'('+rewrite_expression(q[2],edits[n].get(q[1],{}))+')',m[3])
  return m[0][:m.start(3)-m.start()]+body+m[0][m.end(3)-m.start():]
 candidate=CELL.sub(change,original)
 newgraph=graph(candidate);assert set(newgraph)==set(g)
 for n,(kind,pins) in g.items():
  expect={p:[edits.get(n,{}).get(p,{}).get(t,t) for t in ts] for p,ts in pins.items()}
  assert newgraph[n]==(kind,expect),n
 OUT.mkdir(exist_ok=True);table_info=tables(OUT)
 extra=r'''
 wire service_isp_ready,service_isp_valid,service_isp_sof,service_isp_table_valid;
 wire[23:0] service_isp_rgb;wire[15:0] service_isp_id;wire[127:0] service_isp_status;
 wire service_isp_fire;assign service_isp_fire=service_isp_valid && n74389_2;
 wire service_isp_config_write;assign service_isp_config_write=service_host_write && \cpu/host_address ==32'hf9000f28 &&
  (\cpu/host_data [31:24]==8'h49 || \cpu/host_data [31:24]==8'h4a);
 video_input_lut #(.DEFAULT_CRC(32'h__CRC__)) service_isp(
  .clk(process_clk_150),.reset(service_reset),.flush(raw_capture_abort),
  .config_idle(!fixed_boot_run_allowed && service_work_drained && \cpu/boot_terminal ),
  .config_write(service_isp_config_write),.config_data(\cpu/host_data ),
  .s_valid(\raw_frontend/consumer/serializer/second_pending  || |\raw_frontend/consumer/serializer/pair_fifo/queue_count ),
  .s_ready(service_isp_ready),.s_rgb(pair_pixel_rgb),.s_id(pair_pixel_frame_id),
  .s_sof(!\raw_frontend/consumer/serializer/second_pending  && \raw_frontend/consumer/serializer/fifo_data [66]),
  .s_eol(1'b0),.s_eof(1'b0),.m_valid(service_isp_valid),.m_ready(n74389_2),
  .m_rgb(service_isp_rgb),.m_id(service_isp_id),.m_sof(service_isp_sof),.m_eol(),.m_eof(),
  .table_valid(service_isp_table_valid),.status(service_isp_status));
 wire[511:0] service_isp_page;assign service_isp_page={32'h49535031,352'd0,service_isp_status};
 '''.replace('__CRC__',table_info['crc32'])
 # Expose a new typed diagnostic page, keep existing six pages byte-compatible.
 candidate=once(candidate,'if(service_page_phase==5)begin','if(service_page_phase==6)begin')
 candidate=once(candidate,'assign service_selected_page=service_page_phase==4 ?',
  'assign service_selected_page=service_page_phase==6 ? service_isp_page[service_page_part*32+:32] : service_page_phase==4 ?')
 assert candidate.count("8'd5,service_page_sequence")==2
 candidate=candidate.replace("8'd5,service_page_sequence","8'd6,service_page_sequence")
 candidate=once(candidate,"8'd5,service_observation_packet","8'd6,service_observation_packet")
 candidate=once(candidate,'.run_allowed(fixed_boot_run_allowed),.capture_done(cap_done)',
  '.run_allowed(fixed_boot_run_allowed && service_isp_table_valid),.capture_done(cap_done)')
 assert len(re.findall(r'^endmodule\b',candidate,re.M))==1
 declarations,extra=split_net_declarations(extra)
 start=CELL.search(candidate).start();candidate=candidate[:start]+'\n'+'\n'.join(declarations)+'\n'+candidate[start:]
 candidate=re.sub(r'^endmodule\b',lambda m:extra+'\nendmodule',candidate,flags=re.M)
 assert protected_blocks(original)==protected_blocks(candidate)
 (OUT/'native_source.v').write_text(candidate,encoding='utf8')
 for f in BASE.glob('*.sv'):shutil.copy2(f,OUT/f.name)
 shutil.copy2(HERE/'variants/video_service/video_input_lut.sv',OUT/'video_input_lut.sv')
 proof=dict(status='INPUT_ISP_SCOPED_SOURCE_CUT_PASS',parent=str(BASE),parent_sha256=sha(BASE/'native_source.v'),
  candidate_sha256=sha(OUT/'native_source.v'),retained_native_cells=len(g),changed_native_cells=len(edits),
  edits={n:{p:[dict(old=a,new=b) for a,b in aa.items()] for p,aa in pp.items()} for n,pp in edits.items()},
  protected_blocks_byte_equal=True,new_clock_domains=0,tables=table_info,
  hardware_access=False,physical_pass=False,board_release=False,
  pending=['native-cut handshake/marker simulation','RAM mapping/physical/CDC','optical acceptance','RAW rare-stall root cause'],
  inputs={str(p):sha(p) for p in [Path(__file__),HERE/'variants/video_service/video_input_lut.sv',BASE/'native_source.v']})
 save(OUT/'input-isp-edit.json',proof)
 meta=json.loads((BOARD/'inputs.json').read_text());MAP.mkdir();rows=[]
 for row in meta['inputs']:
  old=Path(row['snapshot']);assert sha(old)==row['sha256'];rel=old.relative_to(BOARD)
  origin=OUT/'native_source.v' if rel.as_posix()=='native_source.v' else old
  dst=MAP/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,dst)
  rows.append(dict(source=str(origin),snapshot=str(dst),sha256=sha(dst)))
 for name in ['video_input_lut.sv','isp_r.hex','isp_g.hex','isp_b.hex']:
  shutil.copy2(OUT/name,MAP/name);rows.append(dict(source=str(OUT/name),snapshot=str(MAP/name),sha256=sha(MAP/name)))
 xml=MAP/'ti60f225_oob.xml';s=xml.read_text();s=once(s,'<efx:top_vhdl_arch','<efx:design_file name="video_input_lut.sv" version="default" library="default" />\n<efx:top_vhdl_arch');xml.write_text(s)
 for row in rows:row['sha256']=sha(row['snapshot'])
 shutil.copy2(BOARD/'map.cmd',MAP/'map.cmd')
 save(MAP/'inputs.json',dict(inputs=rows,parent=str(BOARD),native=str(OUT),original_sdc_unchanged=True,
  functional_edit_proof=str(OUT/'input-isp-edit.json'),functional_edit_proof_sha256=sha(OUT/'input-isp-edit.json'),hardware_access=False,board_release=False))
 print(json.dumps({k:proof[k] for k in ['status','candidate_sha256','retained_native_cells','changed_native_cells']}))
if __name__=='__main__':main()
