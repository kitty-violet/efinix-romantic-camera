"""Exact518400/385440-word RAW length classification, without pixel pipeline."""
import argparse,json,os,re,shutil,time
from pathlib import Path
from run_units import ROOT,HERE,run,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 source=ROOT/'tmp/frame-execution/connected-board-video-service-probe58-color-exposure/assembly/raw32_validated_packer.sv'
 template=HERE/'variants/raw_stream_test/tb_raw32_validated_packer.sv'
 shutil.copy2(source,root/source.name)
 s=template.read_text().replace('.WORDS_PER_FRAME(16)', '.WORDS_PER_FRAME(518400)')
 s=s.replace('beat[m_frame_id]!=4', 'beat[m_frame_id]!=129600')
 start=s.index('  for(integer f=0;f<24;f=f+1)');end=s.index('\n end\n initial begin #100000',start)
 s=s[:start]+'''
  begin_frame(1,0);pixels(0,518400);end_frame();
  if(completed!=1 || captured_words!=518400)$fatal(1,"FIRST_FULL_LENGTH");
  begin_frame(0,0);pixels(0,385440);end_frame();
  if(completed!=1 || poisoned!=1 || captured_words!=385440 || !frame_length_error || output_overflow_error)$fatal(1,"SHORT_LENGTH");
  begin_frame(1,1);pixels(1,518400);end_frame();
  if(completed!=2 || poisoned!=1 || m_valid || capture_active || captured_words!=518400 || protocol_error || output_overflow_error)$fatal(1,"LENGTH_RECOVERY");
  $display("PASS RAW_803_EXACT_WORD_COUNT full_words=518400 short_words=385440 completed=%0d cancelled=%0d compared_words=%0d",completed,poisoned,checks);$finish;
'''+s[end:]
 s=s.replace('#100000;', '#50000000;');(root/'tb.sv').write_text(s)
 inp=dict(inputs=[dict(source=str(source),snapshot=str(root/source.name),sha256=sha(source))],
          generated={'tb.sv':sha(root/'tb.sv')},template=str(template),template_sha256=sha(template),
          runner_sha256=sha(__file__),scope='Exact word counts at RAW packer only; not full consumer/DDR/board',host_budget_seconds=45)
 (root/'inputs.json').write_text(json.dumps(inp,indent=2));os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 ms=Path('E:/quatra/questa_fse/win64');start=time.monotonic()
 try:
  run([ms/'vlib.exe','work'],root,'vlib.log',20)
  b,_=run([ms/'vlog.exe','-sv',source.name,'tb.sv'],root,'compile.log',20);assert 'Errors: 0, Warnings: 0' in b
  b,_=run([ms/'vsim.exe','-c','work.tb_raw32_validated_packer','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',45)
  assert 'PASS RAW_803_EXACT_WORD_COUNT' in b and 'Errors: 0, Warnings: 0' in b
  result=dict(status='PASS',line=re.search(r'^# PASS RAW_803.*$',b,re.M).group(0))
 except Exception as e:result=dict(status='FAIL',error=repr(e))
 result.update(seconds=time.monotonic()-start,inputs_sha256=sha(root/'inputs.json'),hardware_access=False,whole_board=False,field_stall_root_cause_proved=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return result['status']=='PASS'

if __name__=='__main__':raise SystemExit(0 if main() else 1)
