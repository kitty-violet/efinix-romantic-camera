"""Same fixed 1280x720 -> 1920x1080 wrapper for scaler area A/B."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--candidate',action='store_true');p.add_argument('--rtl',type=Path);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=HERE.parent/'video_scaler';base=ROOT/'tmp/frame-execution/connected-board-channel-rq1/rtl'
sources=[v/'rgb_scaler_up_smoke_top.sv',base/'resize_axis_up_2_to_3.sv',base/'resize_axis_down_3_to_2.sv',
 HERE/'variants/shared_fixed_scaler/rgb_bilinear_scaler_prototype.sv' if a.candidate else base/'rgb_bilinear_scaler_prototype.sv']
if a.rtl:
 assert not a.candidate and a.rtl.name=='rgb_bilinear_scaler_prototype.sv' and a.rtl.resolve().is_relative_to(HERE/'variants')
 sources[-1]=a.rtl.resolve()
records=[]
for s in sources:
 d=r/s.name;shutil.copy2(s,d);records.append(dict(source=str(s),snapshot=str(d),sha256=sha(d)))
top='rgb_scaler_up_smoke_top'
xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
xml=xml.replace('<efx:design_file name="'+top+'.sv" version="default" library="default" />',''.join('<efx:design_file name="'+s.name+'" version="default" library="default" />' for s in sources))
(r/(top+'.xml')).write_text(xml);(r/(top+'.sdc')).write_text('create_clock -name clk -period 6.667 [get_ports {clk}]\n')
(r/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 240\nexit /b %errorlevel%\n')
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,candidate=a.candidate,runner_sha256=sha(__file__)),indent=2))
body,seconds=run(['cmd.exe','/d','/c','run.cmd'],r,'map.log',timeout=270);assert re.search(r'map\s*:\s*PASS',body)
rpt=r/'outflow'/(top+'.map.rpt');warn=r/'outflow'/(top+'.warn.log')
assert 'EFX-0654' not in warn.read_text()
counts={k:int(n) for k,n in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
result=dict(status='SCALER_ISOLATED_MAP_NOT_BOARD',resources=counts,candidate=a.candidate,seconds=seconds,inputs_sha256=sha(r/'inputs.json'),report_sha256=sha(rpt))
(r/'result.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
