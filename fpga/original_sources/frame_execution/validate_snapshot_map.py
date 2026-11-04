"""Small exact-module synthesis before another full-board build; no USB."""
import argparse,json,re,shutil
from run_units import ROOT,HERE,EF,sha,run
from pathlib import Path
from audit_snapshot_cells import audit
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
source=HERE/'variants/board_control_snapshot/jtag_bringup_snapshot.sv';name='jtag_bringup_snapshot'
shutil.copy2(source,r/source.name)
xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',name)
(r/(name+'.xml')).write_text(xml)
(r/(name+'.sdc')).write_text('create_clock -name system_clk -period 6.666 [get_ports {system_clk}]\ncreate_clock -name tck -period 100 [get_ports {tck}]\n')
(r/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat '+name+'.xml --flow map --timeout 60\nexit /b %errorlevel%\n')
body,seconds=run(['cmd.exe','/d','/c','run.cmd'],r,'map.log',timeout=80)
assert re.search(r'map\s*:\s*PASS',body)
net=r/('outflow/'+name+'.map.v');result=audit(net.read_text())
assert sha(source)==sha(r/source.name)
result.update(seconds=seconds,source_sha256=sha(source),map_sha256=sha(net),hardware_access=False)
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
