"""Measure the qualified dedicated control plane, including real JTAG/CDC."""
import argparse,json,re,shutil,traceback,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,EF,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();proof=a.proof.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists()
g=json.loads((proof/'result.json').read_text());assert g['status']=='PASS_FIXED_SOC_COMPLETE' and sha(proof/'inputs.json')==g['inputs_sha256']
pm=json.loads((proof/'inputs.json').read_text())
for row in pm['inputs']:assert sha(row['snapshot'])==row['sha256']
for name,h in pm['generated'].items():assert sha(proof/name)==h
r.mkdir();top='cnn_fixed_control_soc';records=[]
for name in ('cnn_fixed_control_soc.sv','cnn_boot_sequencer.sv','official_jtag.sv'):
 f=proof/name;d=r/name;shutil.copy2(f,d);records.append(dict(source=str(f),snapshot=str(d),sha256=sha(d)))
ns='http://www.efinixinc.com/enf_proj';ET.register_namespace('efx',ns)
tree=ET.parse(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml');root=tree.getroot();root.set('name',top)
design=root.find('{'+ns+'}design_info');design.find('{'+ns+'}top_module').set('name',top)
for f in list(design.findall('{'+ns+'}design_file')):design.remove(f)
for i,row in enumerate(records,1):design.insert(i,ET.Element('{'+ns+'}design_file',name=Path(row['snapshot']).name,version='default',library='default'))
root.find('{'+ns+'}constraint_info/{'+ns+'}sdc_file').set('name',top+'.sdc')
ET.indent(tree,space=' ');tree.write(r/(top+'.xml'),encoding='utf-8')
(r/(top+'.sdc')).write_text('create_clock -name io_systemClk -period 6.667 [get_ports {io_systemClk}]\ncreate_clock -name jtagCtrl_tck -period 100 [get_ports {jtagCtrl_tck}]\n')
(r/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 180\nexit /b %errorlevel%\n')
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,proof=str(proof),proof_sha256=sha(proof/'result.json'),project_sha256=sha(r/(top+'.xml')),runner_sha256=sha(__file__)),indent=2))
try:
 body,seconds=run(['cmd.exe','/d','/c','run.cmd'],r,'map.log',timeout=210);assert re.search(r'map\s*:\s*PASS',body)
 report=r/'outflow'/(top+'.map.rpt');warning=r/'outflow'/(top+'.warn.log')
 counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',report.read_text(),re.M)}
 assert 'EFX-0654' not in warning.read_text()
 for row in records:assert sha(row['source'])==sha(row['snapshot'])==row['sha256']
 result=dict(status='FIXED_CONTROL_SOC_LOCAL_MAP_PASS',resources=counts,seconds=seconds,inputs_sha256=sha(r/'inputs.json'),report_sha256=sha(report),map_sha256=sha(r/'outflow'/(top+'.map.v')),board_integrated=False,board_fit=False)
 (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
except Exception as e:
 (r/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
