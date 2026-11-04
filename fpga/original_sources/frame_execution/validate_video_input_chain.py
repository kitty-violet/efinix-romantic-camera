"""Bounded camera-color and malformed-frame tests; never accesses hardware.

Uses the frozen 67 assembly, its corrected Bayer RTL, and current capture
prefetch writer. These are interface tests, not full native-design proof.
"""
import argparse,json,shutil,time,re,os
from pathlib import Path
from run_units import ROOT,HERE,run,sha
MS=Path('E:/quatra/questa_fse/win64')

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--kind',choices=['color','recovery','continuous'],required=True);p.add_argument('--short-rows',type=int,default=17);p.add_argument('--wide',action='store_true');p.add_argument('--isp',action='store_true');a=p.parse_args()
 assert not a.isp or a.kind=='continuous'
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 base=ROOT/'tmp/frame-execution/connected-board-video-service-probe58-color-exposure'
 native=ROOT/'tmp/frame-execution/video-service-native67-color-exposure'
 names=['guarded_async_fifo.sv','raw_axi_beat_unpacker.sv','raw_line_bank_ram.sv','raw_line_feeder.sv','legacy_debayer_stream_wrapper.sv','rgb_pair_cdc_serializer.sv','rgb_gain_mul_u8x16_logic.sv','raw_stream_frontend.sv','raw_stream_ingress.sv','raw32_validated_packer.sv']
 files=[base/'assembly'/n for n in names]+[base/'rtl/frame_buffer_V3/DC_FIFO.v']+[base/'rtl/debayer'/n for n in ['debayer_top_2to1.v','rgb_gain_v1.v','line_buffer.v','true_dual_port_ram.v','raw_to_rgb.v']]
 if a.kind in ('recovery','continuous'):
  files += [base/'assembly'/n for n in ['rgb24_stream_pack.sv','axi_frame_write_dma.sv']]+[base/'rtl/rgb_area_downscaler_3to2.sv',HERE/'variants/camera_capture_prefetch/rgb720_stream_writer.sv',ROOT/'tmp/frame-execution/connected-board-fixed-split-bias1/assembly/frame_pair_owner.sv']
 if a.kind=='continuous':
  files=[f for f in files if f.name!='frame_pair_owner.sv']+[HERE/'variants/video_service'/n for n in ['frame_pair_owner.sv','rgb_frame_admission.sv','video_input_lut.sv']]
 records=[]
 for f in files:
  shutil.copy2(f,root/f.name);records.append(dict(path=str(f),sha256=sha(f)))
 assert sha(root/'raw_to_rgb.v')==sha(native/'candidate_raw_to_rgb.v')
 if a.kind=='color':
  # The candidate only changes the pair order at the actual gain input.
  s=(root/'raw_stream_frontend.sv').read_text();s=s.replace('raw_video_consumer','raw_video_consumer_swapped').replace('raw_stream_frontend','raw_stream_frontend_swapped').replace('.raw_pair(rd),.raw_frame_id(rid),.raw_sof(rs),\n','.raw_pair({rd[7:0],rd[15:8]}),.raw_frame_id(rid),.raw_sof(rs),\n')
  assert s.count('.raw_pair({rd[7:0],rd[15:8]}),')==1
  (root/'swapped.sv').write_text(s)
  tb=(HERE/'variants/video_service/tb_video_input_color.sv').read_text();top='tb_video_input_color'
 else:
  tb=(HERE/'variants/raw_stream_test/tb_raw_stream_frontend.sv').read_text();top='tb_raw_stream_frontend'
  tb=tb.replace('WORDS-7,7',f'(W/4)*{a.short_rows},7')
  tb=tb.replace('if(awlen!=0 || pending_w || pending_b || bv)','if(awlen>15 || pending_w || pending_b || bv)').replace('pending_w=1;word_offset','pending_w=awlen+1;word_offset')
  tb=tb.replace('if(!pending_w || !wl || ws!==16\'hffff)','if(!pending_w || wl!==(pending_w==1) || ws!==16\'hffff)')
  tb=tb.replace('pending_w=0;pending_b=3;','word_offset=word_offset+16;pending_w=pending_w-1;if(pending_w==0)pending_b=3;')
  if a.wide:
   tb=tb.replace('W=FULL ? 1920:48','W=1920').replace('FULL ? 1024:64','1024').replace('FULL ? 256:32','256')
   tb=tb.replace('FULL ? 6.734:5.61','6.734').replace('if(FULL)begin if(n%(W/4)==W/4-1)repeat(2200-W/4)','if(1)begin if(n%(W/4)==W/4-1)repeat(1650-W/4)')
   tb=tb.replace('FULL ? 200000000:5000000','6500000')
   tb=tb.replace('integer prefix_ns=0;', '''initial forever begin
    #200000;
    $display("PROGRESS ns=%0d done=%0d failed=%0d aborts=%0d phase=%d req/ack=%b%b rawfv/fr=%b%b poison=%b unpack=%b%b feeder=%d/full%b reserve=%d rgb=%b%b down=%b%b capbusy=%b dma=%d packstate=%d",$time,completions,failures,aborts,dut.ingress.phase,dut.request,dut.ack,dut.ingress.fv,dut.ingress.fr,dut.ingress.poison,dut.consumer.uv,dut.consumer.ur,dut.consumer.feeder.state,dut.consumer.feeder.bank_full,dut.consumer.serializer.reserved_pairs,mv,mr,dv,dr,capture_busy,writer.dma.state,dut.ingress.packer.state);$fflush();
   end
   integer prefix_ns=0;''')
  tb=tb.replace('$fatal(1,"TIMEOUT complete=%d failed=%d aborts=%d",completions,failures,aborts);','$display("STALL phase=%d ack=%b abort=%b capbusy=%b rawv/r=%b%b downv/r=%b%b writeractive=%b dma=%d",dut.ingress.phase,dut.ack,abort_capture,capture_busy,mv,mr,dv,dr,writer.active,writer.dma.state);$fatal(1,"TIMEOUT complete=%d failed=%d aborts=%d",completions,failures,aborts);')
  if a.kind=='continuous':
   from validate_video_input_lut import tables
   tables(root)
   # Camera candidate pair order applies to both stream and numerical reference.
   f=root/'raw_stream_frontend.sv';s=f.read_text();s=s.replace('.raw_pair(rd),.raw_frame_id(rid),.raw_sof(rs),\n','.raw_pair({rd[7:0],rd[15:8]}),.raw_frame_id(rid),.raw_sof(rs),\n');f.write_text(s)
   gate='''
 wire gv,gr,gs,gl,ge,passing;wire[23:0] grgb;wire[15:0] gid;
 wire[31:0] admitted,skipped,orphans;
 rgb_frame_admission gate(.clk(clk),.reset(reset),.abort_frame(abort_capture),.allow_capture(1'b1),.slot_ready(cap_ready),
 .s_valid(dv),.s_ready(dr),.s_rgb(d_rgb),.s_frame_id(did),.s_sof(ds),.s_eol(dl),.s_eof(de),
 .m_valid(gv),.m_ready(gr),.m_rgb(grgb),.m_frame_id(gid),.m_sof(gs),.m_eol(gl),.m_eof(ge),
 .admitted_frames(admitted),.skipped_frames(skipped),.orphan_pixels(orphans),.passing(passing));
 '''
   tb=tb.replace(' wire req,cap_ready,done,failed,writer_error,owner_error;',' wire req,cap_ready,done,failed,writer_error,owner_error;'+gate)
   tb=tb.replace('.s_valid(dv),.s_ready(dr),.s_rgb(d_rgb ^ 24\'h808080),.s_frame_id(did),.s_sof(ds),.s_eol(dl),.s_eof(de)',".s_valid(gv),.s_ready(gr),.s_rgb(grgb ^ 24'h808080),.s_frame_id(gid),.s_sof(gs),.s_eol(gl),.s_eof(ge)")
   tb=tb.replace('frame_pair_owner owner(.clk(clk),.reset(reset),','frame_pair_owner owner(.clk(clk),.reset(reset),.bypass_compute(1\'b1),')
   tb=tb.replace('if(failed)begin if(good[current_id])$fatal(1,"GOOD_FRAME_FAILED");failures=failures+1;end','if(failed)begin failures=failures+1;end')
   start=tb.index('  good_frame(0);');end=tb.index('\n end\n initial begin #(FULL',start)
   tb=tb[:start]+'''
  // No wait for capture completion between frames. Overlap camera, malformed
  // frame cancellation, DDR stall and retained-front display ownership.
  for(integer f=0;f<15;f++)begin
   good[f]=(f%4!=2);block_memory=(f==4 || f==8);
   camera_send(f,good[f]?WORDS:WORDS-7,7);
   repeat(80)@(negedge cam);
  end
  block_memory=0;repeat(20000)@(negedge cam);
  if(capture_busy || abort_capture || dut.ingress.phase!=0 || passing)$fatal(1,"CONTINUOUS_RECOVERY_STUCK");
  good[15]=1;camera_send(15,WORDS,7);
  wait(commit_time[15]!=0);
  for(integer f=0;f<16;f++)if(commit_time[f]!=0)begin ref_frame_send(f);compare_frame(f);end
  if(completions<4 || failures<1 || aborts<1 || aw_count!=b_count)$fatal(1,"CONTINUOUS_COVERAGE");
  if(admitted!=completions+failures)$fatal(1,"ACCOUNTING admitted=%d complete=%d failed=%d",admitted,completions,failures);
  $display("PASS CONTINUOUS_RAW_CURRENT_SERVICE admitted=%0d completed=%0d failed=%0d skipped=%0d raw_abort=%0d compared_bytes=%0d",admitted,completions,failures,skipped,aborts,compared);$finish;
 '''+tb[end:]
   # This stress validates current admission/owner and pair-order correction;
   # ISP itself has separate numeric/flush/backpressure tests, not inserted here.
 if a.isp:
  # Stateful candidate precedes the scaler. The complete-frame reference uses
  # an independent combinational lookup (no shared valid/ready implementation).
  insert='''
 wire iv,ir,isof,ieol,ieof;wire[23:0] irgb;wire[15:0] iid;wire table_ok;
 video_input_lut #(.DEFAULT_CRC(32'he483a10c)) isp(.clk(clk),.reset(reset),.flush(abort_capture),
 .config_idle(1'b0),.config_write(1'b0),.config_data(32'd0),
 .s_valid(mv),.s_ready(mr),.s_rgb(rgb),.s_id(mid),.s_sof(ms),.s_eol(ml),.s_eof(me),
 .m_valid(iv),.m_ready(ir),.m_rgb(irgb),.m_id(iid),.m_sof(isof),.m_eol(ieol),.m_eof(ieof),
 .table_valid(table_ok),.status());
 reg[7:0] gamma_table[0:255];initial $readmemh("isp_r.hex",gamma_table);
 '''
  tb=tb.replace(' wire dv,dr,ds,dl,de,down_error;',insert+'\n wire dv,dr,ds,dl,de,down_error;')
  tb=tb.replace(' wire rdv,rds,rdl,rde,ref_down_error;', ' wire[23:0] reference_color;assign reference_color={gamma_table[rrgb[7:0]],gamma_table[rrgb[15:8]],gamma_table[rrgb[23:16]]};\n wire rdv,rds,rdl,rde,ref_down_error;')
  old='.s_valid(mv && !abort_capture),.s_ready(mr),.s_rgb(rgb),.s_frame_id(mid),.s_sof(ms),.s_eol(ml)'
  assert tb.count(old)==1
  tb=tb.replace(old,'.s_valid(iv && !abort_capture),.s_ready(ir),.s_rgb(irgb),.s_frame_id(iid),.s_sof(isof),.s_eol(ieol)')
  assert tb.count('.s_rgb(rrgb)')==1
  tb=tb.replace('.s_rgb(rrgb)','.s_rgb(reference_color)').replace('PASS CONTINUOUS_RAW_CURRENT_SERVICE','PASS CONTINUOUS_RAW_ISP_CURRENT_SERVICE')
 (root/'tb.sv').write_text(tb)
 inputs=dict(files=records,generated={f.name:sha(f) for f in root.glob('*.sv')},runner_sha256=sha(__file__),scope='67 RTL adjacent interfaces; not native binding/physical/board proof')
 (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
 os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 start=time.monotonic()
 try:
  run([MS/'vlib.exe','work'],root,'vlib.log',20)
  run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[f.name for f in root.iterdir() if f.suffix in ('.sv','.v')]],root,'compile.log',40)
  body,seconds=run([MS/'vsim.exe','-c','work.'+top,'-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',90 if a.wide else 60)
  lines=re.findall(r'^# (?:PASS|COLOR|FRONTEND_FRAME).*$',body,re.M);assert any('PASS' in l for l in lines)
  result=dict(status='PASS',seconds=time.monotonic()-start,lines=lines,inputs_sha256=sha(root/'inputs.json'),hardware_access=False)
 except Exception as e:result=dict(status='FAIL',error=str(e),seconds=time.monotonic()-start)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return result['status']=='PASS'

if __name__=='__main__':raise SystemExit(0 if main() else 1)
