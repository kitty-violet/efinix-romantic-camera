"""Compare native CPU W payload ownership under the same adapter map wrapper."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--candidate',action='store_true');a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
src=HERE/'variants/fixed_cpu_payload/sapphire_axi32_to_axi128.sv' if a.candidate else HERE.parent/'ddr_fabric/sapphire_axi32_to_axi128.sv'
dst=r/src.name;shutil.copy2(src,dst);top=src.stem
xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
(r/(top+'.xml')).write_text(xml);(r/(top+'.sdc')).write_text('create_clock -name clk -period 6.667 [get_ports {clk}]\n')
(r/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 180\nexit /b %errorlevel%\n')
(r/'inputs.json').write_text(json.dumps(dict(source=str(src),snapshot=str(dst),sha256=sha(dst),candidate=a.candidate,runner_sha256=sha(__file__)),indent=2))
body,seconds=run(['cmd.exe','/d','/c','run.cmd'],r,'map.log',timeout=210);assert re.search(r'map\s*:\s*PASS',body)
rpt=r/'outflow'/(top+'.map.rpt');warn=r/'outflow'/(top+'.warn.log')
assert 'EFX-0654' not in warn.read_text();assert sha(src)==sha(dst)
counts={k:int(n) for k,n in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
result=dict(status='CPU_ADAPTER_MAP_NOT_BOARD',resources=counts,candidate=a.candidate,seconds=seconds,inputs_sha256=sha(r/'inputs.json'),report_sha256=sha(rpt))
(r/'result.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
