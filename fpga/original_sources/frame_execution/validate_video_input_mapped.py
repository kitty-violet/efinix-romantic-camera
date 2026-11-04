"""Map the complete ISP with the official mapper and simulate its actual primitives."""
import argparse,json,os,re,shutil,time,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,HERE,EF,sha,run
from validate_video_input_lut import tables

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 source=HERE/'variants/video_service/video_input_lut.sv';shutil.copy2(source,root/source.name);tables(root)
 base=ROOT/'tmp/frame-execution/connected-board-video-service-probe58-color-exposure'
 ns={'e':'http://www.efinixinc.com/enf_proj'};ET.register_namespace('efx',ns['e'])
 tree=ET.parse(base/'ti60f225_oob.xml');project=tree.getroot();design=project.find('e:design_info',ns)
 for child in list(design):design.remove(child)
 ET.SubElement(design,'{'+ns['e']+'}top_module',name='video_input_lut')
 ET.SubElement(design,'{'+ns['e']+'}design_file',name=source.name,version='default',library='default')
 # No interface/bitstream flow is invoked; the same Ti60 target and map settings.
 tree.write(root/'ti60f225_oob.xml',encoding='utf-8');shutil.copy2(base/'map.cmd',root/'map.cmd')
 evidence={str(source):sha(source),str(root/'ti60f225_oob.xml'):sha(root/'ti60f225_oob.xml')}
 (root/'inputs.json').write_text(json.dumps(dict(evidence=evidence,runner_sha256=sha(__file__)),indent=2))
 start=time.monotonic();os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';ms=Path('E:/quatra/questa_fse/win64')
 try:
  run(['cmd.exe','/d','/c','map.cmd'],root,'map.log',120)
  mapped=root/'outflow/ti60f225_oob.map.v';text=mapped.read_text()
  warns=(root/'outflow/ti60f225_oob.warn.log').read_text();assert not re.search(r'does not have a driver|non-driven|cannot be initialized',warns)
  tb=(HERE/'variants/video_service/tb_video_input_lut.sv').read_text().replace('dut.count!=0','status[41:32]!=0')
  (root/'tb.sv').write_text(tb)
  models=[]
  for kind in sorted(set(re.findall(r'^\s*(EFX_\w+)\s',text,re.M))):
   f=EF/'sim_models/verilog'/(kind.lower()+'.v');assert f.exists(),kind
   shutil.copy2(f,root/f.name);models.append(f.name)
  run([ms/'vlib.exe','work'],root,'vlib.log',20)
  run([ms/'vlog.exe','-sv','-timescale','1ns/1ps',*models,mapped,'tb.sv'],root,'compile.log',40)
  body,_=run([ms/'vsim.exe','-c','work.tb_video_input_lut','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',60)
  assert 'PASS INPUT_LUT' in body
  result=dict(status='PASS',map_sha256=sha(mapped),lines=re.findall(r'^# PASS.*$',body,re.M),official_models=models)
 except Exception as e:result=dict(status='FAIL',error=str(e))
 result.update(seconds=time.monotonic()-start,inputs_sha256=sha(root/'inputs.json'),hardware_access=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return result['status']=='PASS'
if __name__=='__main__':raise SystemExit(0 if main() else 1)
