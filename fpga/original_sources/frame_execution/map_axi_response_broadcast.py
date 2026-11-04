"""Same-board fabric map screen; no change in protocol, frequency or device."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--candidate',action='store_true');p.add_argument('--rtl',type=Path);a=p.parse_args()
assert not a.rtl or a.candidate
r=a.root.resolve();base=a.base.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
meta=json.loads((base/'inputs.json').read_text());br=json.loads((base/'result.json').read_text());assert sha(base/'inputs.json')==br['inputs_sha256']
names=('axi_three_master_burst_arbiter.sv','sapphire_axi32_to_axi128.sv','ti60_ddr_four_master_fabric.sv')
records=[]
for name in (*names,'ddr_memory_map.vh'):
 src=base/('' if name.endswith('.vh') else 'assembly')/name
 assert any(x['snapshot']==str(src) and x['sha256']==sha(src) for x in meta['inputs'])
 if a.candidate and name==names[0]:src=HERE/'variants/axi_response_broadcast'/name
 if a.rtl and name==names[0]:
  src=a.rtl.resolve();assert src.is_relative_to(HERE/'variants') and src.name==name
 dst=r/name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(src)))
top='ti60_ddr_four_master_fabric'
xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
old=f'<efx:design_file name="{top}.sv" version="default" library="default" />';assert old in xml
xml=xml.replace(old,'\n'.join(f'<efx:design_file name="{n}" version="default" library="default" />' for n in names))
(r/(top+'.xml')).write_text(xml);(r/(top+'.sdc')).write_text('create_clock -name clk -period 6.667 [get_ports {clk}]\n')
(r/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 120\nexit /b %errorlevel%\n')
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__),base=str(base),candidate=a.candidate,project_sha256=sha(r/(top+'.xml'))),indent=2)+'\n')
body,seconds=run(['cmd.exe','/d','/c','run.cmd'],r,'map.log',timeout=150);assert re.search(r'map\s*:\s*PASS',body)
rpt=r/'outflow'/(top+'.map.rpt');warn=r/'outflow'/(top+'.warn.log');assert 'EFX-0654' not in warn.read_text()
counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
for x in records:assert sha(x['source'])==x['sha256']==sha(x['snapshot'])
result=dict(status='AXI_RESPONSE_FABRIC_MAP_NOT_BOARD',resources=counts,seconds=seconds,inputs_sha256=sha(r/'inputs.json'),candidate=a.candidate,report_sha256=sha(rpt),map_sha256=sha(r/'outflow'/(top+'.map.v')))
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
