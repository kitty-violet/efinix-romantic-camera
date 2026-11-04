"""Freeze the qualified streaming RAW front end into a measured complete board.
This changes no pin, PHY, PLL, timing constraint, CNN contract or trained weight.
The result is an offline capacity candidate, not a programming release.
"""
import argparse,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,HERE,sha
def once(text,old,new):
 assert text.count(old)==1,(old[:90],text.count(old))
 return text.replace(old,new)
def replace_between(text,start,end,new):
 assert text.count(start)==text.count(end)==1,(start,end)
 i=text.index(start);j=text.index(end,i)
 return text[:i]+new+text[j:]
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();base=a.base.resolve();proof=a.proof.resolve()
 assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 meta=json.loads((base/'inputs.json').read_text());mapped=json.loads((base/'result.json').read_text())
 assert sha(base/'inputs.json')==mapped['inputs_sha256']
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
 qualified=json.loads((proof/'result.json').read_text());assert qualified['status']=='RAW_STREAM_FRONTEND_PASS'
 assert qualified['inputs_sha256']==sha(proof/'inputs.json') and len(qualified['results'])==3
 tested={Path(x['source']).name:x for x in json.loads((proof/'inputs.json').read_text())['inputs']}
 added=list((HERE/'variants/raw_stream').glob('*.sv'))
 for f in added:assert f.name in tested and sha(f)==tested[f.name]['sha256']
 root.mkdir()
 for folder in ('rtl','assembly'):shutil.copytree(base/folder,root/folder)
 for f in base.iterdir():
  if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,root/f.name)
 top=root/'rtl/ti60f225_oob_top.v';t=top.read_text(encoding='utf-8')
 t=replace_between(t,'wire raw_beat_valid;','\n`endif\n\n\nwire [7:0] ddr516_bid', '''wire raw_capture_abort,raw_capture_busy;
wire [10:0] raw_stream_errors;
wire [31:0] raw_aborted_frames;
assign s_axi_awlock=1'b0;assign s_axi_awcache=0;assign s_axi_awprot=0;
assign s_axi_arlock=1'b0;assign s_axi_arcache=0;assign s_axi_arprot=0;
assign raw_axi_arid=0;assign raw_axi_araddr=0;assign raw_axi_arlen=0;
assign raw_axi_arsize=3'd4;assign raw_axi_arburst=2'd1;assign raw_axi_arvalid=0;assign raw_axi_rready=1;
assign raw_axi_awid=0;assign raw_axi_awaddr=0;assign raw_axi_awlen=0;
assign raw_axi_awsize=3'd4;assign raw_axi_awburst=2'd1;assign raw_axi_awvalid=0;
assign raw_axi_wdata=0;assign raw_axi_wstrb=0;assign raw_axi_wlast=0;assign raw_axi_wvalid=0;assign raw_axi_bready=1;
assign camera_write_fault=(|raw_stream_errors[10:7]) | camera_boundary_error;
assign camera_read_fault=1'b0;
assign camera_line_fault=|raw_stream_errors[6:0];
assign ch0_r=0;assign ch0_g=0;assign ch0_b=0;assign ch0_vs=0;assign ch0_hs=0;assign ch0_de=0;
''')
 t=replace_between(t,'wire debayer_pair_valid;','reg [3:0] hdmi_reset_pipe', '''wire pair_pixel_valid;
wire [23:0] pair_pixel_rgb;
wire [15:0] pair_pixel_frame_id;
wire pair_pixel_sof,pair_pixel_eol,pair_pixel_eof,hdmi_source_ready;
raw_stream_frontend raw_frontend(
 .reset(!vid_rst_n || !w_arstn),.camera_clk(i_mipi_rx_pclk),.line_clk(vid_clk_dvi2),.pixel_clk(process_clk_150),
 .frame_start(camera_frame_start),.frame_end(camera_frame_end),.pixel_valid(camera_pixel_valid),
 .pixel_word({w_mipi_rx_data[39:32],w_mipi_rx_data[29:22],w_mipi_rx_data[19:12],w_mipi_rx_data[9:2]}),
 .r_gain(|r_gain ? r_gain : 16'd255),.g_gain(|g_gain ? g_gain : 16'd255),.b_gain(|b_gain ? b_gain : 16'd255),
 .m_valid(pair_pixel_valid),.m_ready(hdmi_source_ready),.m_rgb(pair_pixel_rgb),.m_frame_id(pair_pixel_frame_id),
 .m_sof(pair_pixel_sof),.m_eol(pair_pixel_eol),.m_eof(pair_pixel_eof),
 .capture_busy(raw_capture_busy),.capture_abort(raw_capture_abort),.errors(raw_stream_errors),.aborted_frames(raw_aborted_frames));

''')
 t=once(t,'.cancel(control_cancel),.enable(control_enable && cal_done_sync)', '.capture_abort(raw_capture_abort),.capture_busy(raw_capture_busy),\n    .cancel(control_cancel),.enable(control_enable && cal_done_sync)')
 top.write_text(t,encoding='utf-8')
 video=root/'assembly/cnn_video_pipeline.sv';t=video.read_text()
 t=once(t,'input wire cancel,enable,compare_mode,','input wire cancel,enable,compare_mode,\n    input wire capture_abort,output wire capture_busy,')
 t=once(t,'assign s_ready = process_rst_n && down_input_ready;', 'assign s_ready = process_rst_n && !capture_abort && down_input_ready;')
 t=once(t,'.clk(process_clk), .rst_n(process_rst_n),\n        .s_valid(s_valid && process_rst_n)', '.clk(process_clk), .rst_n(process_rst_n && !capture_abort),\n        .s_valid(s_valid && process_rst_n && !capture_abort)')
 t=once(t,'.clk(process_clk),.reset(!process_rst_n),.cancel(cancel),', '.clk(process_clk),.reset(!process_rst_n),.capture_abort(capture_abort),.capture_busy_out(capture_busy),.cancel(cancel),')
 video.write_text(t)
 pipe=root/'assembly/cnn_rgb_frame_pipeline.sv';t=pipe.read_text()
 t=once(t,'input wire clk,reset,cancel,enable,compare_mode,','input wire clk,reset,cancel,enable,compare_mode,\n input wire capture_abort,output wire capture_busy_out,')
 t=once(t,'rgb720_ddr_writer #(.MAX_CAPTURE_BURST(MAX_CAPTURE_BURST)) capture(', 'assign capture_busy_out=capture_busy;\n rgb720_stream_writer #(.MAX_CAPTURE_BURST(MAX_CAPTURE_BURST)) capture(.frame_abort(capture_abort),')
 t=replace_between(t,' wire [1:0] cap_awready,cap_wready,cap_bvalid,cap_arready,cap_rlast,cap_rvalid;',' ti60_ddr_four_master_fabric #(', ''' // Fixed video-only acquisition has one RGB writer. No second RAW master.
 assign capture_merge_error=1'b0;
 assign cam_arid=0;assign cam_araddr=0;assign cam_arlen=0;assign cam_arsize=3'd4;assign cam_arburst=2'd1;assign cam_arvalid=0;assign cam_rready=1;
 assign cam_awid=0;assign cam_awaddr=rgb_awaddr;assign cam_awlen=rgb_awlen;assign cam_awsize=3'd4;assign cam_awburst=2'd1;assign cam_awvalid=rgb_awvalid;
 assign rgb_awready=cam_awready;assign cam_wdata=rgb_wdata;assign cam_wstrb=rgb_wstrb;assign cam_wlast=rgb_wlast;assign cam_wvalid=rgb_wvalid;
 assign rgb_wready=cam_wready;assign rgb_bresp=cam_bresp;assign rgb_bvalid=cam_bvalid;assign cam_bready=rgb_bready;
 assign raw_axi_arready=0;assign raw_axi_rid=0;assign raw_axi_rdata=0;assign raw_axi_rresp=0;assign raw_axi_rlast=0;assign raw_axi_rvalid=0;
 assign raw_axi_awready=0;assign raw_axi_wready=0;assign raw_axi_bid=0;assign raw_axi_bresp=0;assign raw_axi_bvalid=0;
''')
 pipe.write_text(t)
 changed={top,video,pipe};records=[]
 for row in meta['inputs']:
  dst=root/Path(row['snapshot']).relative_to(base)
  records.append(dict(source=str(row['snapshot']),snapshot=str(dst),sha256=sha(dst),base_sha256=row['sha256'],transformed=dst in changed))
 paths=[root/Path(f).relative_to(base) for f in meta['source_files']]
 for f in added:
  dst=root/'assembly'/f.name;shutil.copy2(f,dst);paths.append(dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(dst)))
 ns={'e':'http://www.efinixinc.com/enf_proj'};ET.register_namespace('efx',ns['e'])
 xml=root/'ti60f225_oob.xml';tree=ET.parse(xml);design=tree.getroot().find('e:design_info',ns)
 for e in list(design.findall('e:design_file',ns)):design.remove(e)
 for i,f in enumerate(paths,1):design.insert(i,ET.Element('{'+ns['e']+'}design_file',name=f.relative_to(root).as_posix(),version='default',library='default'))
 ET.indent(tree,space='    ');tree.write(xml,encoding='utf-8',xml_declaration=False)
 for row in records:
  if Path(row['snapshot'])==xml:row.update(sha256=sha(xml),transformed=True)
 meta.update(inputs=records,base=str(base),base_inputs_sha256=sha(base/'inputs.json'),source_files=list(map(str,paths)),raw_frontend_proof_sha256=sha(proof/'result.json'),runner_sha256=sha(__file__),board_release=False,raw_fullframe_ddr_removed=True,capture_abort_recoverable=True)
 (root/'inputs.json').write_text(json.dumps(meta,indent=2)+'\n')
 for row in records:assert sha(row['snapshot'])==row['sha256']
 print('RAW_STREAM_BOARD_FROZEN',len(records))
if __name__=='__main__':main()
