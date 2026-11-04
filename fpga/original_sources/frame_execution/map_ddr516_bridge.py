"""Check that the public burst buffer becomes block RAM before a whole-board map."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,EF,sha,run

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--bridge',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(exist_ok=False)
 src=a.bridge.resolve();top='ddr516_axi_address_bridge';dst=root/(top+'.sv');shutil.copy2(src,dst)
 xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
 (root/(top+'.xml')).write_text(xml);(root/(top+'.sdc')).write_text('create_clock -name clk -period 6.6667 [get_ports {clk}]\n')
 (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 600\nexit /b %errorlevel%\n')
 data=dict(source=str(src),source_sha256=sha(src),runner_sha256=sha(__file__),files={f.name:sha(f) for f in root.iterdir()})
 (root/'inputs.json').write_text(json.dumps(data,indent=2))
 body,seconds=run(['cmd.exe','/d','/c','run.cmd'],root,'map.log',timeout=660);assert re.search(r'map\s*:\s*PASS',body)
 rpt=root/'outflow'/(top+'.map.rpt');net=root/'outflow'/(top+'.map.v')
 counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
 assert sha(dst)==data['source_sha256']
 passed=counts.get('RAM10',0)+counts.get('DPRAM10',0)>=4 and counts.get('FF',0)<1000
 result=dict(status='BURST_BUFFER_RAM_MAP_PASS' if passed else 'BURST_BUFFER_RAM_MAP_FAIL',resources=counts,seconds=seconds,
  source_sha256=sha(dst),inputs_sha256=sha(root/'inputs.json'),report_sha256=sha(rpt),map_sha256=sha(net),full_board=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 assert passed,result
if __name__=='__main__':main()
