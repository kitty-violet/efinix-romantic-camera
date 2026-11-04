"""Map a frozen source selector; compare candidates with the same top/SDC."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,EF,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();source=a.source.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
assert source.name=='fixed_source_columns6.sv';root.mkdir();top=source.stem;shutil.copy2(source,root/source.name)
xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
(root/(top+'.xml')).write_text(xml);(root/(top+'.sdc')).write_text('create_clock -name clk -period 6.667 [get_ports {clk}]\n')
(root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 120\nexit /b %errorlevel%\n')
meta=dict(source=str(source),source_sha256=sha(source),runner_sha256=sha(__file__),files={f.name:sha(f) for f in root.iterdir()})
(root/'inputs.json').write_text(json.dumps(meta,indent=2)+'\n')
body,seconds=run(['cmd.exe','/d','/c','run.cmd'],root,'map.log',timeout=150);assert re.search(r'map\s*:\s*PASS',body)
rpt=root/'outflow'/(top+'.map.rpt');net=root/'outflow'/(top+'.map.v');warn=root/'outflow'/(top+'.warn.log')
counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
assert 'EFX-0654' not in warn.read_text() and sha(source)==sha(root/source.name)==meta['source_sha256']
result=dict(status='SOURCE_COLUMN_MAP_ONLY',resources=counts,seconds=seconds,inputs_sha256=sha(root/'inputs.json'),map_sha256=sha(net),report_sha256=sha(rpt),full_board=False,physical_pass=False)
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
