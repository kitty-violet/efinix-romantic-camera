"""Analysis-only constant projection of the verified CNN interface; no RTL edits."""
import argparse,collections,json,re,shutil
from pathlib import Path
from run_units import ROOT,EF,sha,run

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
reference=ROOT/'tmp/frame-execution/joint-fourway-map3'
manifest=json.loads((reference/'inputs.json').read_text());records=[]
for s in manifest['inputs']:
 src=Path(s['snapshot']);assert sha(src)==s['sha256'];dst=root/src.name;shutil.copy2(src,dst)
 records.append(dict(path=str(src),snapshot=str(dst),sha256=sha(dst)))
core=(root/'compute_column_cluster.sv').read_text();header=core[:core.index('\n);')+3]
header=header[header.index('module compute_column_cluster('):]
declarations=re.split(r'\b(?:input|output|inout)\b',header[header.index('(')+1:header.rindex(')')])[1:]
ports=[]
for d in declarations:
 d=re.sub(r'\[[^]]+\]','',d);d=re.sub(r'\b(?:wire|reg|signed)\b','',d)
 ports.extend(re.findall(r'\b[A-Za-z_][A-Za-z_0-9]*\b',d))
assert len(set(ports))==len(ports) and 'cfg_residual' in ports and 'pc_residual' in ports
constants={'cfg_residual':"1'b0",'cfg_row_deltas':"66'd0",'pc_residual':"1'b0"}
wrapper=header.replace('module compute_column_cluster(','module production_projection(').replace('output reg','output wire')
wrapper+='\n compute_column_cluster dut(\n'+',\n'.join('  .'+name+'('+constants.get(name,name)+')' for name in ports)+'\n );\nendmodule\n'
(root/'production_projection.sv').write_text(wrapper)
xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter','production_projection')
xml=xml.replace('<efx:design_file name="production_projection.sv" version="default" library="default" />',
 ''.join('<efx:design_file name="'+Path(s['snapshot']).name+'" version="default" library="default" />' for s in records)+'<efx:design_file name="production_projection.sv" version="default" library="default" />')
xml=xml.replace('</efx:synthesis>','<efx:param name="max_threads" value="2" value_type="e_integer" /></efx:synthesis>')
(root/'production_projection.xml').write_text(xml)
(root/'production_projection.sdc').write_text('create_clock -name clk -period 6.6667 [get_ports {clk}]\n')
(root/'inputs.json').write_text(json.dumps(dict(records=records,constants=constants,wrapper_sha256=sha(root/'production_projection.sv'),runner_sha256=sha(__file__)),indent=2))
(root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat production_projection.xml --flow map --timeout 900\nexit /b %errorlevel%\n')
body,seconds=run(['cmd.exe','/d','/c','run.cmd'],root,'map.log');assert re.search(r'map\s*:\s*PASS',body)
net=root/'outflow/production_projection.map.v';rpt=root/'outflow/production_projection.map.rpt'
counts=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
assert counts.get('DSP24')==256 and counts.get('DSP48')==32
for s in records:assert sha(s['snapshot'])==s['sha256']
result=dict(status='ANALYSIS_CONSTANT_PROJECTION_MAP_NOT_FUNCTIONAL_GATE',resources=counts,constants=constants,seconds=seconds,
 source_top='same frozen joint-fourway-map3 core, only three public ports tied as in full-frame TB',
 inputs_sha256=sha(root/'inputs.json'),report_sha256=sha(rpt),map_sha256=sha(net),pnr=False,full_system=False,new_timing_measurement=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
