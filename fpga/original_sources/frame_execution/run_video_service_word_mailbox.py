"""Exercise all sixteen asynchronous word selections and a stopped source."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir()
 old=HERE/'variants/video_service/tb_video_debug_chunks.sv'
 text=old.read_text()
 changes=[('[1:0]','[3:0]'),('[127:0]','[31:0]'),('.WIDTH(128)','.WIDTH(32)'),('p*128+:128','p*32+:32'),('p<4','p<16'),
 ('sequence_id!=4','sequence_id!=16'),('returned_part!=3','returned_part!=15'),('sequence_id!=5','sequence_id!=17'),
 ('repeat(4)@(negedge dst_clk);','for(p=0;p<16;p=p+1)source_data[p*32+:32]=32\'h12340000+p;\n  repeat(4)@(negedge dst_clk);'),
 ('four_parts','sixteen_parts')]
 for a,b in changes:
  assert a in text,a
  text=text.replace(a,b)
 tb=out/'tb_video_debug_chunks.sv';tb.write_text(text)
 src=HERE/'variants/video_service_word/video_debug_mailbox.sv';shutil.copy2(src,out/src.name)
 inputs={str(f):sha(f) for f in [Path(__file__),old,src,tb,out/src.name]}
 run([MS/'vlib.exe','work'],out,'vlib.log',20)
 run([MS/'vlog.exe','-sv',tb,out/src.name],out,'compile.log',30)
 body,secs=run([MS/'vsim.exe','-c','work.tb_video_debug_chunks','-do','onerror {quit -code 1}; run -all; quit -code 0'],out,'simulation.log',30)
 assert 'PASS VIDEO_DEBUG_CHUNKS sixteen_parts' in body
 assert all(sha(f)==h for f,h in inputs.items())
 result=dict(status='WORD_MAILBOX_ALL_PARTS_STOPPED_CLOCK_LATE_ACK_PASS',inputs=inputs,seconds=secs,hardware_access=False,board_release=False)
 (out/'result.json').write_text(json.dumps(result,indent=2));print(result['status'])
if __name__=='__main__':main()
